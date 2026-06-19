#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// 地名のメタデータ。Lua 側の分類情報と座標を保持する。
struct PlaceInfo {
    std::string name;
    std::string japanese;
    std::string macroGroup;
    std::string subGroup;
    double lat = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
};

// TSV から読み取った地域ごとの語彙・規則性特徴量をまとめる。
struct RegionData {
    std::string name;
    double lat = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, std::vector<std::string>> lexicalTokens;
    std::unordered_set<std::string> lexicalFeatures;
    std::unordered_set<std::string> regularityFeatures;
};

// 学習・評価に使う 1 地域ぶんの特徴ベクトル。
struct TrainingSample {
    std::string name;
    std::string macroGroup;
    std::string subGroup;
    double lat = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
    std::unordered_set<std::string> lexicalFeatures;
    std::unordered_set<std::string> regularityFeatures;
};

// 同じ方言群に属する学習サンプル群から作る代表プロトタイプ。
struct Prototype {
    std::string macroGroup;
    std::string subGroup;
    std::vector<const TrainingSample*> members;
    std::unordered_map<std::string, double> lexicalFrequency;
    std::unordered_map<std::string, double> regularityFrequency;
    double meanLat = std::numeric_limits<double>::quiet_NaN();
    double meanLon = std::numeric_limits<double>::quiet_NaN();
};

// 分類結果と、各スコア成分の内訳。
struct ClassificationResult {
    std::string name;
    std::string macroGroup;
    std::string subGroup;
    double score = -1.0;
    double lexicalScore = 0.0;
    double regularityScore = 0.0;
    double geographyScore = 0.0;
};

struct TsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

struct TeeStream {
    std::ostream& console;
    std::ofstream file;

    TeeStream(const std::string& path, std::ostream& consoleStream)
        : console(consoleStream), file(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary) {
        if (!file) {
            throw std::runtime_error("failed to open output file: " + path);
        }
    }

    template <typename T>
    TeeStream& operator<<(const T& value) {
        console << value;
        file << value;
        return *this;
    }

    TeeStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        manip(console);
        manip(file);
        return *this;
    }

    TeeStream& operator<<(std::ios& (*manip)(std::ios&)) {
        manip(console);
        manip(file);
        return *this;
    }

    TeeStream& operator<<(std::ios_base& (*manip)(std::ios_base&)) {
        manip(console);
        manip(file);
        return *this;
    }
};

// TSV の語彙データが各地域でどれくらい埋まっているかの統計。
struct CoverageStats {
    int lexicalRowCount = 0;
    std::unordered_map<std::string, int> nonEmptyCellsByRegion;
};

struct RegionSummaryRow {
    std::string region;
    std::string goldMacroGroup;
    std::string goldSubGroup;
    std::string predictedMacroGroup;
    std::string predictedSubGroup;
    double lat = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
    int nonEmptyCells = 0;
    int lexicalFeatureCount = 0;
    int regularityFeatureCount = 0;
    double score = -1.0;
    double lexicalScore = 0.0;
    double regularityScore = 0.0;
    double geographyScore = 0.0;
    std::string matchKind;
};

struct PlaceLuaOrder {
    std::vector<std::string> names;
    std::unordered_set<std::string> seen;
};

std::string trim(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first])) != 0) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1])) != 0) {
        --last;
    }
    return s.substr(first, last - first);
}

std::string toLowerAscii(std::string s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    }
    return s;
}

std::string formatDouble(double value, int precision) {
    if (!std::isfinite(value)) {
        return std::string();
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string formatRegionLabel(const RegionData& region) {
    std::ostringstream oss;
    oss << region.name << " (" << region.lexicalFeatures.size() << ")";
    return oss.str();
}

bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

std::vector<std::string> splitTsvLine(const std::string& line) {
    std::vector<std::string> cols;
    std::string current;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\t') {
            cols.push_back(current);
            current.clear();
        } else {
            current.push_back(line[i]);
        }
    }
    cols.push_back(current);
    return cols;
}

// 単純な TSV ローダー。先頭行をヘッダ、それ以降をデータ行として読む。
TsvTable readTsv(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open TSV: " + path);
    }

    TsvTable table;
    std::string line;
    bool firstLine = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (firstLine) {
            if (line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line.erase(0, 3);
            }
            firstLine = false;
        }
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> cols = splitTsvLine(line);
        if (table.header.empty()) {
            table.header = cols;
        } else {
            table.rows.push_back(cols);
        }
    }
    return table;
}

// トークン正規化の前処理として、文字列置換をまとめて行う。
std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return s;
    }
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string removeParenthetical(const std::string& s) {
    std::string out;
    int roundDepth = 0;
    int squareDepth = 0;
    int curlyDepth = 0;
    int angleDepth = 0;
    int cornerDepth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        if (ch == '(') {
            ++roundDepth;
            continue;
        }
        if (ch == ')') {
            if (roundDepth > 0) {
                --roundDepth;
            }
            continue;
        }
        if (ch == '[') {
            ++squareDepth;
            continue;
        }
        if (ch == ']') {
            if (squareDepth > 0) {
                --squareDepth;
            }
            continue;
        }
        if (ch == '{') {
            ++curlyDepth;
            continue;
        }
        if (ch == '}') {
            if (curlyDepth > 0) {
                --curlyDepth;
            }
            continue;
        }
        if (ch == '<') {
            ++angleDepth;
            continue;
        }
        if (ch == '>') {
            if (angleDepth > 0) {
                --angleDepth;
            }
            continue;
        }
        if (static_cast<unsigned char>(ch) == 0xE3) {
            if (i + 2 < s.size()) {
                const unsigned char b1 = static_cast<unsigned char>(s[i + 1]);
                const unsigned char b2 = static_cast<unsigned char>(s[i + 2]);
                if (b1 == 0x80 && b2 == 0x8A) { // 《
                    ++cornerDepth;
                    i += 2;
                    continue;
                }
                if (b1 == 0x80 && b2 == 0x8B) { // 》
                    if (cornerDepth > 0) {
                        --cornerDepth;
                    }
                    i += 2;
                    continue;
                }
                if (b1 == 0x80 && b2 == 0x94) { // 〔
                    ++cornerDepth;
                    i += 2;
                    continue;
                }
                if (b1 == 0x80 && b2 == 0x95) { // 〕
                    if (cornerDepth > 0) {
                        --cornerDepth;
                    }
                    i += 2;
                    continue;
                }
            }
        }
        if (roundDepth == 0 && squareDepth == 0 && curlyDepth == 0 && angleDepth == 0 && cornerDepth == 0) {
            out.push_back(ch);
        }
    }
    return out;
}

// セル内の語形候補を簡易トークン化し、特徴量として使える形にそろえる。
std::vector<std::string> tokenizeField(const std::string& raw) {
    std::string s = raw;
    s = replaceAll(s, "<br>", " ");
    s = replaceAll(s, "<br/>", " ");
    s = replaceAll(s, "<br />", " ");
    s = removeParenthetical(s);

    for (std::size_t i = 0; i < s.size(); ++i) {
        char& ch = s[i];
        if (ch == ',' || ch == ';' || ch == '|' || ch == ':' || ch == '"' || ch == '?' || ch == '!' || ch == '=') {
            ch = ' ';
        }
    }

    std::vector<std::string> tokens;
    std::string current;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char uch = static_cast<unsigned char>(s[i]);
        const bool asciiLetter = std::isalpha(uch) != 0;
        const bool extra = s[i] == '-' || s[i] == '_' || s[i] == '\'' || uch >= 0x80;
        if (asciiLetter || extra) {
            current.push_back(s[i]);
        } else {
            std::string token = trim(current);
            current.clear();
            if (!token.empty()) {
                token = toLowerAscii(token);
                bool ok = false;
                for (std::size_t j = 0; j < token.size(); ++j) {
                    const char c = token[j];
                    if ((c >= 'a' && c <= 'z') || c == '-' || c == '_' || static_cast<unsigned char>(c) >= 0x80) {
                        ok = true;
                        break;
                    }
                }
                if (ok) {
                    tokens.push_back(token);
                }
            }
        }
    }
    std::string token = trim(current);
    if (!token.empty()) {
        token = toLowerAscii(token);
        tokens.push_back(token);
    }
    return tokens;
}

bool containsAnyToken(const std::vector<std::string>& values, const std::vector<std::string>& keys) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t j = 0; j < keys.size(); ++j) {
            if (values[i] == keys[j]) {
                return true;
            }
        }
    }
    return false;
}

double deg2rad(double deg) {
    return deg * 3.14159265358979323846 / 180.0;
}

double haversineKm(double lat1, double lon1, double lat2, double lon2) {
    const double dLat = deg2rad(lat2 - lat1);
    const double dLon = deg2rad(lon2 - lon1);
    const double a = std::pow(std::sin(dLat / 2.0), 2.0) +
        std::cos(deg2rad(lat1)) * std::cos(deg2rad(lat2)) * std::pow(std::sin(dLon / 2.0), 2.0);
    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    return 6371.0 * c;
}

double jaccardSimilarity(const std::unordered_set<std::string>& left, const std::unordered_set<std::string>& right) {
    if (left.empty() && right.empty()) {
        return 1.0;
    }
    if (left.empty() || right.empty()) {
        return 0.0;
    }

    const std::unordered_set<std::string>* smaller = &left;
    const std::unordered_set<std::string>* larger = &right;
    if (left.size() > right.size()) {
        smaller = &right;
        larger = &left;
    }

    std::size_t intersection = 0;
    for (std::unordered_set<std::string>::const_iterator it = smaller->begin(); it != smaller->end(); ++it) {
        if (larger->find(*it) != larger->end()) {
            ++intersection;
        }
    }
    const std::size_t unionSize = left.size() + right.size() - intersection;
    if (unionSize == 0) {
        return 1.0;
    }
    return static_cast<double>(intersection) / static_cast<double>(unionSize);
}

double pairwiseGeographicSimilarity(const RegionData& left, const RegionData& right) {
    if (!std::isfinite(left.lat) || !std::isfinite(left.lon) || !std::isfinite(right.lat) || !std::isfinite(right.lon)) {
        return 0.0;
    }
    const double d = haversineKm(left.lat, left.lon, right.lat, right.lon);
    return 1.0 / (1.0 + d / 250.0);
}

std::unordered_set<std::string> toTokenSet(const std::vector<std::string>& values) {
    std::unordered_set<std::string> tokenSet;
    for (std::size_t i = 0; i < values.size(); ++i) {
        tokenSet.insert(values[i]);
    }
    return tokenSet;
}

bool isAsciiVowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

std::string normalizeComparisonToken(const std::string& token) {
    std::string out;
    out.reserve(token.size());
    for (std::size_t i = 0; i < token.size(); ++i) {
        const char c = token[i];
        if (c == '\'') {
            continue;
        }
        out.push_back(c);
    }

    if (out.size() >= 2 && out[0] == 'h' && isAsciiVowel(out[1])) {
        out.erase(0, 1);
    }

    for (std::size_t i = 1; i < out.size(); ++i) {
        if (out[i] == 'k' && isAsciiVowel(out[i - 1])) {
            const bool atWordEnd = (i + 1 == out.size());
            const bool beforeConsonant = (i + 1 < out.size() && !isAsciiVowel(out[i + 1]));
            if (atWordEnd || beforeConsonant) {
                out[i] = 'h';
            }
        }
    }

    return out;
}

std::unordered_set<std::string> toNormalizedTokenSet(const std::vector<std::string>& values) {
    std::unordered_set<std::string> tokenSet;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::string normalized = normalizeComparisonToken(values[i]);
        if (!normalized.empty()) {
            tokenSet.insert(normalized);
        }
    }
    return tokenSet;
}

double lexicalGlossSimilarity(const std::vector<std::string>& leftTokens, const std::vector<std::string>& rightTokens) {
    const std::unordered_set<std::string> leftSet = toNormalizedTokenSet(leftTokens);
    const std::unordered_set<std::string> rightSet = toNormalizedTokenSet(rightTokens);
    if (leftSet.empty() && rightSet.empty()) {
        return 1.0;
    }
    if (leftSet.empty() || rightSet.empty()) {
        return 0.0;
    }

    const std::unordered_set<std::string>* smaller = &leftSet;
    const std::unordered_set<std::string>* larger = &rightSet;
    if (leftSet.size() > rightSet.size()) {
        smaller = &rightSet;
        larger = &leftSet;
    }

    for (std::unordered_set<std::string>::const_iterator it = smaller->begin(); it != smaller->end(); ++it) {
        if (larger->find(*it) != larger->end()) {
            return 1.0;
        }
    }

    return 0.0;
}

double shrinkSimilarity(double rawScore, std::size_t observationCount, double priorMean, double priorStrength) {
    const double n = static_cast<double>(observationCount);
    if (n <= 0.0) {
        return priorMean;
    }
    return (n * rawScore + priorStrength * priorMean) / (n + priorStrength);
}

double pairwiseLexicalSimilarity(const RegionData& left, const RegionData& right) {
    double similaritySum = 0.0;
    std::size_t overlapCount = 0;

    const std::unordered_map<std::string, std::vector<std::string>>* smaller = &left.lexicalTokens;
    const std::unordered_map<std::string, std::vector<std::string>>* larger = &right.lexicalTokens;
    if (left.lexicalTokens.size() > right.lexicalTokens.size()) {
        smaller = &right.lexicalTokens;
        larger = &left.lexicalTokens;
    }

    for (std::unordered_map<std::string, std::vector<std::string>>::const_iterator it = smaller->begin(); it != smaller->end(); ++it) {
        std::unordered_map<std::string, std::vector<std::string>>::const_iterator jt = larger->find(it->first);
        if (jt == larger->end()) {
            continue;
        }
        similaritySum += lexicalGlossSimilarity(it->second, jt->second);
        ++overlapCount;
    }

    const double rawLexical = overlapCount == 0 ? 0.5 : similaritySum / static_cast<double>(overlapCount);
    return shrinkSimilarity(rawLexical, overlapCount, 0.5, 8.0);
}

double pairwiseRegularitySimilarity(const RegionData& left, const RegionData& right) {
    std::size_t overlapGlossCount = 0;

    const std::unordered_map<std::string, std::vector<std::string>>* smaller = &left.lexicalTokens;
    const std::unordered_map<std::string, std::vector<std::string>>* larger = &right.lexicalTokens;
    if (left.lexicalTokens.size() > right.lexicalTokens.size()) {
        smaller = &right.lexicalTokens;
        larger = &left.lexicalTokens;
    }

    for (std::unordered_map<std::string, std::vector<std::string>>::const_iterator it = smaller->begin(); it != smaller->end(); ++it) {
        if (larger->find(it->first) != larger->end()) {
            ++overlapGlossCount;
        }
    }

    double rawRegularity = 0.0;
    if (left.regularityFeatures.empty() && right.regularityFeatures.empty()) {
        rawRegularity = 1.0;
    } else if (!left.regularityFeatures.empty() && !right.regularityFeatures.empty()) {
        const std::unordered_set<std::string>* smaller = &left.regularityFeatures;
        const std::unordered_set<std::string>* larger = &right.regularityFeatures;
        if (left.regularityFeatures.size() > right.regularityFeatures.size()) {
            smaller = &right.regularityFeatures;
            larger = &left.regularityFeatures;
        }

        bool shared = false;
        for (std::unordered_set<std::string>::const_iterator it = smaller->begin(); it != smaller->end(); ++it) {
            if (larger->find(*it) != larger->end()) {
                shared = true;
                break;
            }
        }
        rawRegularity = shared ? 1.0 : 0.0;
    }

    return shrinkSimilarity(rawRegularity, overlapGlossCount, 0.5, 8.0);
}

double pairwiseSimilarity(const RegionData& left, const RegionData& right) {
    if (left.name == right.name) {
        return 1.0;
    }
    const double lexical = pairwiseLexicalSimilarity(left, right);
    const double regularity = pairwiseRegularitySimilarity(left, right);
    const double geography = pairwiseGeographicSimilarity(left, right);
    return lexical * 0.60 + regularity * 0.30 + geography * 0.10;
}

std::ofstream openUtf8Tsv(const std::string& path) {
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open TSV output file: " + path);
    }
    const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
    out.write(reinterpret_cast<const char*>(bom), 3);
    return out;
}

std::vector<std::string> collectSortedRegionNames(const std::unordered_map<std::string, RegionData>& regions) {
    std::vector<std::string> names;
    for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        names.push_back(it->first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> collectOrderedRegionNames(
    const std::unordered_map<std::string, RegionData>& regions,
    const std::vector<std::string>& preferredOrder) {

    std::vector<std::string> names;
    std::unordered_set<std::string> added;

    for (std::size_t i = 0; i < preferredOrder.size(); ++i) {
        if (regions.find(preferredOrder[i]) != regions.end() && added.insert(preferredOrder[i]).second) {
            names.push_back(preferredOrder[i]);
        }
    }

    std::vector<std::string> remaining;
    for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        if (added.find(it->first) == added.end()) {
            remaining.push_back(it->first);
        }
    }
    std::sort(remaining.begin(), remaining.end());
    names.insert(names.end(), remaining.begin(), remaining.end());
    return names;
}

std::unordered_map<std::string, RegionData> filterRegionsByLexicalFeatureCount(
    const std::unordered_map<std::string, RegionData>& regions,
    std::size_t minimumLexicalFeatureCount) {

    std::unordered_map<std::string, RegionData> filtered;
    for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        if (it->second.lexicalFeatures.size() >= minimumLexicalFeatureCount) {
            filtered[it->first] = it->second;
        }
    }
    return filtered;
}

void writeSimilarityMatrixTsv(
    const std::string& path,
    const std::unordered_map<std::string, RegionData>& regions,
    const std::vector<std::string>& preferredOrder) {

    std::ofstream out = openUtf8Tsv(path);
    const std::vector<std::string> names = collectOrderedRegionNames(regions, preferredOrder);

    out << "region";
    for (std::size_t i = 0; i < names.size(); ++i) {
        const RegionData& region = regions.find(names[i])->second;
        out << '\t' << formatRegionLabel(region);
    }
    out << '\n';

    for (std::size_t row = 0; row < names.size(); ++row) {
        const RegionData& left = regions.find(names[row])->second;
        out << formatRegionLabel(left);
        for (std::size_t col = 0; col < names.size(); ++col) {
            const RegionData& right = regions.find(names[col])->second;
            out << '\t' << formatDouble(pairwiseSimilarity(left, right), 6);
        }
        out << '\n';
    }
}

void writeDistanceMatrixTsv(
    const std::string& path,
    const std::unordered_map<std::string, RegionData>& regions,
    const std::vector<std::string>& preferredOrder) {

    std::ofstream out = openUtf8Tsv(path);
    const std::vector<std::string> names = collectOrderedRegionNames(regions, preferredOrder);

    out << "region";
    for (std::size_t i = 0; i < names.size(); ++i) {
        const RegionData& region = regions.find(names[i])->second;
        out << '\t' << formatRegionLabel(region);
    }
    out << '\n';

    for (std::size_t row = 0; row < names.size(); ++row) {
        const RegionData& left = regions.find(names[row])->second;
        out << formatRegionLabel(left);
        for (std::size_t col = 0; col < names.size(); ++col) {
            const RegionData& right = regions.find(names[col])->second;
            const double distance = 1.0 - pairwiseSimilarity(left, right);
            out << '\t' << formatDouble(distance, 6);
        }
        out << '\n';
    }
}

void writeRegionSummaryTsv(const std::string& path, const std::vector<RegionSummaryRow>& rows) {
    std::ofstream out = openUtf8Tsv(path);
    out << "region\tgold_macro_group\tgold_sub_group\tpredicted_macro_group\tpredicted_sub_group"
        << "\tlat\tlon\tnon_empty_cells\tlexical_features\tregularity_features"
        << "\tscore\tlexical_score\tregularity_score\tgeography_score\tmatch_kind\n";

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const RegionSummaryRow& row = rows[i];
        out << row.region << '\t'
            << row.goldMacroGroup << '\t'
            << row.goldSubGroup << '\t'
            << row.predictedMacroGroup << '\t'
            << row.predictedSubGroup << '\t'
            << formatDouble(row.lat, 6) << '\t'
            << formatDouble(row.lon, 6) << '\t'
            << row.nonEmptyCells << '\t'
            << row.lexicalFeatureCount << '\t'
            << row.regularityFeatureCount << '\t'
            << formatDouble(row.score, 6) << '\t'
            << formatDouble(row.lexicalScore, 6) << '\t'
            << formatDouble(row.regularityScore, 6) << '\t'
            << formatDouble(row.geographyScore, 6) << '\t'
            << row.matchKind << '\n';
    }
}

std::string normalizePlaceKey(const std::string& s);

PlaceLuaOrder parsePlaceLuaOrder(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open Lua: " + path);
    }

    PlaceLuaOrder order;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string t = trim(line);
        if (!startsWith(t, "{ name = \"")) {
            continue;
        }

        const std::size_t begin = 10;
        const std::size_t end = t.find('"', begin);
        if (end == std::string::npos) {
            continue;
        }

        const std::string englishName = trim(t.substr(begin, end - begin));
        const std::size_t japanesePos = t.find("japanese = \"");
        std::string orderedName = englishName;
        if (japanesePos != std::string::npos) {
            const std::size_t japaneseBegin = japanesePos + 12;
            const std::size_t japaneseEnd = t.find('"', japaneseBegin);
            if (japaneseEnd != std::string::npos) {
                orderedName = normalizePlaceKey(t.substr(japaneseBegin, japaneseEnd - japaneseBegin));
            }
        } else {
            orderedName = normalizePlaceKey(englishName);
        }

        if (!orderedName.empty() && order.seen.insert(orderedName).second) {
            order.names.push_back(orderedName);
        }
    }

    return order;
}

// `place.lua` から地域名・分類ラベル・座標を読み取り、検索用マップを作る。
std::unordered_map<std::string, PlaceInfo> parsePlaceLua(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open Lua: " + path);
    }

    std::unordered_map<std::string, PlaceInfo> places;
    std::string line;
    std::string currentMacro;
    std::string currentSub;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string t = trim(line);

        if (t.find("name = \"") != std::string::npos && t.find("{ name = ") == std::string::npos) {
            const std::size_t pos = t.find("name = \"");
            if (pos != std::string::npos) {
                const std::size_t begin = pos + 8;
                const std::size_t end = t.find('"', begin);
                if (end != std::string::npos) {
                    const std::string value = t.substr(begin, end - begin);
                    if (value == "Hokkaido" || value == "Sakhalin" || value == "Kuril" || value == "Honshu") {
                        currentMacro = value;
                    } else if (!currentMacro.empty()) {
                        currentSub = value;
                    }
                }
            }
        }

        if (startsWith(t, "{ name = \"")) {
            auto readStringField = [&](const std::string& key) -> std::string {
                const std::size_t pos = t.find(key);
                if (pos == std::string::npos) {
                    return std::string();
                }
                const std::size_t begin = pos + key.size();
                const std::size_t end = t.find('"', begin);
                if (end == std::string::npos) {
                    return std::string();
                }
                return t.substr(begin, end - begin);
            };
            auto readNumberField = [&](const std::string& key) -> double {
                const std::size_t pos = t.find(key);
                if (pos == std::string::npos) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const std::size_t begin = pos + key.size();
                std::size_t end = begin;
                while (end < t.size() && (std::isdigit(static_cast<unsigned char>(t[end])) || t[end] == '.' || t[end] == '-')) {
                    ++end;
                }
                return std::atof(t.substr(begin, end - begin).c_str());
            };

            PlaceInfo info;
            info.name = readStringField("{ name = \"");
            info.japanese = normalizePlaceKey(readStringField("japanese = \""));
            info.lat = readNumberField("lat = ");
            info.lon = readNumberField("long = ");
            info.macroGroup = currentMacro;
            info.subGroup = currentSub;
            if (!info.name.empty()) {
                places[info.name] = info;
                if (!info.japanese.empty()) {
                    places[info.japanese] = info;
                }
            }
        }
    }

    return places;
}

std::string normalizePlaceKey(const std::string& s) {
    std::string out = trim(s);
    if (out.size() >= 3 &&
        static_cast<unsigned char>(out[0]) == 0xEF &&
        static_cast<unsigned char>(out[1]) == 0xBB &&
        static_cast<unsigned char>(out[2]) == 0xBF) {
        out.erase(0, 3);
    }
    out = replaceAll(out, " ", "");
    out = replaceAll(out, "\t", "");
    out = replaceAll(out, "\r", "");
    out = replaceAll(out, "\n", "");
    return out;
}

void setupConsoleUtf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

// `place.tsv` から地域ごとの語彙特徴量と座標を読み取る。
// 空白セルがある場合、地理的に最も近い非空白の地域からデータを補完する。
std::unordered_map<std::string, RegionData> parsePlaceTsv(const std::string& path) {
    TsvTable table = readTsv(path);
    if (table.header.size() < 4) {
        throw std::runtime_error("unexpected TSV header");
    }

    // 列インデックスから正規化済み地域名へのマッピングを作る。
    std::vector<std::string> colNames(table.header.size());
    for (std::size_t col = 3; col < table.header.size(); ++col) {
        colNames[col] = normalizePlaceKey(table.header[col]);
    }

    std::unordered_map<std::string, RegionData> regions;
    for (std::size_t col = 3; col < table.header.size(); ++col) {
        RegionData region;
        region.name = colNames[col];
        regions[region.name] = region;
    }

    // まず座標行を読み取る。
    for (std::size_t row = 0; row < table.rows.size(); ++row) {
        const std::vector<std::string>& values = table.rows[row];
        if (values.empty()) {
            continue;
        }
        const std::string rowKey = trim(values[0]);
        if (rowKey == "lat") {
            for (std::size_t col = 3; col < table.header.size() && col < values.size(); ++col) {
                const std::string v = trim(values[col]);
                if (!v.empty()) {
                    regions[colNames[col]].lat = std::atof(v.c_str());
                }
            }
        }
        if (rowKey == "lon") {
            for (std::size_t col = 3; col < table.header.size() && col < values.size(); ++col) {
                const std::string v = trim(values[col]);
                if (!v.empty()) {
                    regions[colNames[col]].lon = std::atof(v.c_str());
                }
            }
        }
    }

    // 語彙行を処理する。空白セルは最寄りの非空白地域から補完する。
    for (std::size_t row = 0; row < table.rows.size(); ++row) {
        const std::vector<std::string>& values = table.rows[row];
        if (values.empty()) {
            continue;
        }
        const std::string rowKey = trim(values[0]);
        if (rowKey == "lat" || rowKey == "lon" || rowKey == "en") {
            continue;
        }

        const std::string gloss = toLowerAscii(rowKey);

        // この行で非空白のセルを持つ列のインデックスを集める。
        std::vector<std::size_t> nonEmptyCols;
        for (std::size_t col = 3; col < table.header.size(); ++col) {
            const std::string cell = (col < values.size()) ? trim(values[col]) : std::string();
            if (!cell.empty()) {
                nonEmptyCols.push_back(col);
            }
        }

        for (std::size_t col = 3; col < table.header.size(); ++col) {
            const std::string regionName = colNames[col];
            std::string cell = (col < values.size()) ? values[col] : std::string();

            // 空白セル補完は一時的に無効化。
            // if (trim(cell).empty() && !nonEmptyCols.empty()) {
            //     const RegionData& target = regions[regionName];
            //     if (std::isfinite(target.lat) && std::isfinite(target.lon)) {
            //         double bestDist = std::numeric_limits<double>::max();
            //         std::size_t bestCol = nonEmptyCols[0];
            //         for (std::size_t k = 0; k < nonEmptyCols.size(); ++k) {
            //             const RegionData& candidate = regions[colNames[nonEmptyCols[k]]];
            //             if (std::isfinite(candidate.lat) && std::isfinite(candidate.lon)) {
            //                 const double d = haversineKm(target.lat, target.lon, candidate.lat, candidate.lon);
            //                 if (d < bestDist) {
            //                     bestDist = d;
            //                     bestCol = nonEmptyCols[k];
            //                 }
            //             }
            //         }
            //         cell = (bestCol < values.size()) ? values[bestCol] : std::string();
            //     }
            // }

            std::vector<std::string> tokens = tokenizeField(cell);
            if (tokens.empty()) {
                continue;
            }
            RegionData& region = regions[regionName];
            std::vector<std::string>& dest = region.lexicalTokens[gloss];
            dest.insert(dest.end(), tokens.begin(), tokens.end());
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                region.lexicalFeatures.insert(gloss + "=" + tokens[i]);
            }
        }
    }

    return regions;
}

// 語彙セルの埋まり具合を集計し、特徴抽出量の妥当性確認に使う。
CoverageStats collectCoverageStats(const std::string& path) {
    TsvTable table = readTsv(path);
    CoverageStats stats;
    if (table.header.size() < 4) {
        return stats;
    }

    for (std::size_t col = 3; col < table.header.size(); ++col) {
        stats.nonEmptyCellsByRegion[normalizePlaceKey(table.header[col])] = 0;
    }

    for (std::size_t row = 0; row < table.rows.size(); ++row) {
        const std::vector<std::string>& values = table.rows[row];
        if (values.empty()) {
            continue;
        }
        const std::string rowKey = trim(values[0]);
        if (rowKey == "lat" || rowKey == "lon" || rowKey == "en") {
            continue;
        }

        ++stats.lexicalRowCount;
        for (std::size_t col = 3; col < table.header.size(); ++col) {
            const std::string regionName = normalizePlaceKey(table.header[col]);
            const std::string cell = (col < values.size()) ? trim(values[col]) : std::string();
            if (!cell.empty()) {
                ++stats.nonEmptyCellsByRegion[regionName];
            }
        }
    }

    return stats;
}

void addRegularityFeature(std::unordered_set<std::string>& features, const std::string& name) {
    features.insert(name);
}

// 語彙トークンから、既知の音韻・形態的な規則性特徴を導出する。
void extractRegularityFeatures(RegionData& region) {
    const auto& lex = region.lexicalTokens;
    auto has = [&](const std::string& gloss, const std::vector<std::string>& forms) -> bool {
        std::unordered_map<std::string, std::vector<std::string>>::const_iterator it = lex.find(gloss);
        return it != lex.end() && containsAnyToken(it->second, forms);
    };

    if (has("milk", {"toope"}) || has("year", {"paa"}) || has("to say", {"yee"}) || has("name", {"ree", "tee"}) || has("to drink", {"kuu"})) {
        addRegularityFeature(region.regularityFeatures, "REG_V1V1_V1");
    }
    if (has("ash (or ashes)", {"uuna", "uyna"}) || has("here [adv.]", {"teeta", "teyta"})) {
        addRegularityFeature(region.regularityFeatures, "REG_V1V1_V1Y_V1");
    }
    if (has("we / us (「除外形」は相手を含まない我々, exclusive. 「包括形」は相手を含む我々, inclusive.)", {"ciokay", "anokay", "aokay", "coka", "anoka", "aoka"}) ||
        has("i / me", {"anokay", "anoka", "ciokay", "cokay"}) ||
        has("ye, you (all)", {"eciokay", "ecookay", "ecioka", "ecokay", "esokay"})) {
        addRegularityFeature(region.regularityFeatures, "REG_V1Y_V1");
    }
    if (has("fish", {"ciep"}) || has("i / me", {"kuani", "kani"})) {
        addRegularityFeature(region.regularityFeatures, "REG_V1V2_V2");
    }
    if (has("to kill", {"rayki", "tayki"}) || has("to hit [to beat]", {"sitayki"}) || has("to pierce it [to stab it]", {"otki"})) {
        addRegularityFeature(region.regularityFeatures, "REG_C1E_C1I");
    }
    if (has("to wash it", {"uray"}) || has("to push", {"optuy"}) || has("to dig (it)", {"poi"}) || has("belly, stomack", {"pse"})) {
        addRegularityFeature(region.regularityFeatures, "REG_C1V_C1_OR_YV1");
    }
    if (has("man, male", {"okkaw"}) || has("to cut it", {"tuwe"}) || has("guts, intestines", {"tuworop"}) || has("rain", {"ruwanpe"})) {
        addRegularityFeature(region.regularityFeatures, "REG_Y_W");
    }
    if (has("star", {"nocuy"}) || has("to pierce it [to stab it]", {"cuy"})) {
        addRegularityFeature(region.regularityFeatures, "REG_IW_UY");
    }
    if (has("to wash it", {"huraye"}) || has("husband", {"hoku", "hoko"}) || has("child, children [in general]", {"hekaci", "hekattar"}) || has("up", {"herekas", "herikasi", "herikas"})) {
        addRegularityFeature(region.regularityFeatures, "REG_H_ZERO");
    }
    if (has("claw", {"ham"}) || has("sand", {"hota"}) || has("fog", {"hurar"}) || has("flower", {"hapappo"})) {
        addRegularityFeature(region.regularityFeatures, "REG_ZERO_H");
    }
    if (has("to walk", {"akkas", "ahkas"}) || has("to lie down", {"hokke"}) || has("rain", {"atto", "ahto"})) {
        addRegularityFeature(region.regularityFeatures, "REG_C1C2_C2C2_HC2");
    }
    if (has("to see, to look at<br> to be seen", {"nukara"}) || has("to sleep", {"mokoro"}) || has("to be new", {"asiri"}) || has("to be good", {"pirika"}) || has("worm<br> insect<br> bug", {"kikiri"}) || has("fog", {"uurara"})) {
        addRegularityFeature(region.regularityFeatures, "REG_RV_R");
    }

    if (has("fire", {"unci"})) {
        addRegularityFeature(region.regularityFeatures, "GROUP_SAKHALIN_FIRE_UNCI");
    }
    if (has("water", {"wahka"})) {
        addRegularityFeature(region.regularityFeatures, "SUPPORT_WAHKA");
    }
    if (has("path, road", {"tuu"}) || has("name", {"tee"})) {
        addRegularityFeature(region.regularityFeatures, "SUPPORT_EDGE_FORM");
    }
}

// ラベル付き地域だけを学習サンプルに変換する。Honshu は学習対象から除外する。
std::vector<TrainingSample> buildTrainingSamples(
    const std::unordered_map<std::string, RegionData>& regions,
    const std::unordered_map<std::string, PlaceInfo>& places) {

    std::vector<TrainingSample> samples;
    for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        std::unordered_map<std::string, PlaceInfo>::const_iterator pit = places.find(normalizePlaceKey(it->first));
        if (pit == places.end()) {
            continue;
        }
        if (pit->second.macroGroup == "Honshu") {
            continue;
        }

        TrainingSample sample;
        sample.name = it->first;
        sample.macroGroup = pit->second.macroGroup;
        sample.subGroup = pit->second.subGroup;
        sample.lat = it->second.lat;
        sample.lon = it->second.lon;
        sample.lexicalFeatures = it->second.lexicalFeatures;
        sample.regularityFeatures = it->second.regularityFeatures;
        samples.push_back(sample);
    }
    return samples;
}

// 同一マクロ群・サブ群ごとに、平均的な特徴頻度と重心座標を持つプロトタイプを作る。
std::vector<Prototype> buildPrototypes(const std::vector<TrainingSample>& samples, const std::string& excludedName) {
    std::map<std::pair<std::string, std::string>, Prototype> grouped;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!excludedName.empty() && samples[i].name == excludedName) {
            continue;
        }
        std::pair<std::string, std::string> key(samples[i].macroGroup, samples[i].subGroup);
        Prototype& proto = grouped[key];
        proto.macroGroup = samples[i].macroGroup;
        proto.subGroup = samples[i].subGroup;
        proto.members.push_back(&samples[i]);
    }

    std::vector<Prototype> prototypes;
    for (std::map<std::pair<std::string, std::string>, Prototype>::iterator it = grouped.begin(); it != grouped.end(); ++it) {
        Prototype proto = it->second;
        double latSum = 0.0;
        double lonSum = 0.0;
        int geoCount = 0;

        for (std::size_t i = 0; i < proto.members.size(); ++i) {
            const TrainingSample& s = *proto.members[i];
            for (std::unordered_set<std::string>::const_iterator fit = s.lexicalFeatures.begin(); fit != s.lexicalFeatures.end(); ++fit) {
                proto.lexicalFrequency[*fit] += 1.0;
            }
            for (std::unordered_set<std::string>::const_iterator fit = s.regularityFeatures.begin(); fit != s.regularityFeatures.end(); ++fit) {
                proto.regularityFrequency[*fit] += 1.0;
            }
            if (std::isfinite(s.lat) && std::isfinite(s.lon)) {
                latSum += s.lat;
                lonSum += s.lon;
                ++geoCount;
            }
        }

        const double memberCount = static_cast<double>(proto.members.size());
        for (std::unordered_map<std::string, double>::iterator fit = proto.lexicalFrequency.begin(); fit != proto.lexicalFrequency.end(); ++fit) {
            fit->second /= memberCount;
        }
        for (std::unordered_map<std::string, double>::iterator fit = proto.regularityFrequency.begin(); fit != proto.regularityFrequency.end(); ++fit) {
            fit->second /= memberCount;
        }
        if (geoCount > 0) {
            proto.meanLat = latSum / static_cast<double>(geoCount);
            proto.meanLon = lonSum / static_cast<double>(geoCount);
        }
        prototypes.push_back(proto);
    }

    return prototypes;
}

// サンプル側の特徴がプロトタイプにどれだけ含まれるかで一致度を測る。
double scoreFeatureSet(
    const std::unordered_set<std::string>& sampleFeatures,
    const std::unordered_map<std::string, double>& prototypeFeatures) {

    if (sampleFeatures.empty() || prototypeFeatures.empty()) {
        return 0.0;
    }

    double score = 0.0;
    for (std::unordered_set<std::string>::const_iterator it = sampleFeatures.begin(); it != sampleFeatures.end(); ++it) {
        std::unordered_map<std::string, double>::const_iterator pit = prototypeFeatures.find(*it);
        if (pit != prototypeFeatures.end()) {
            score += pit->second;
        }
    }
    return score / static_cast<double>(sampleFeatures.size());
}

// 地理的距離をゆるやかな減衰スコアに変換する。
double scoreGeography(double lat, double lon, const Prototype& proto) {
    if (!std::isfinite(lat) || !std::isfinite(lon) || !std::isfinite(proto.meanLat) || !std::isfinite(proto.meanLon)) {
        return 0.0;
    }
    const double d = haversineKm(lat, lon, proto.meanLat, proto.meanLon);
    return 1.0 / (1.0 + d / 250.0);
}

// 語彙・規則性・地理の各スコアを合成し、最良の方言群を返す。
ClassificationResult classifySample(const TrainingSample& sample, const std::vector<Prototype>& prototypes) {
    ClassificationResult best;
    best.name = sample.name;

    for (std::size_t i = 0; i < prototypes.size(); ++i) {
        const Prototype& proto = prototypes[i];
        const double lexicalScore = scoreFeatureSet(sample.lexicalFeatures, proto.lexicalFrequency);
        const double regularityScore = scoreFeatureSet(sample.regularityFeatures, proto.regularityFrequency);
        const double geographyScore = scoreGeography(sample.lat, sample.lon, proto);
        const double totalScore = lexicalScore * 0.45 + regularityScore * 0.40 + geographyScore * 0.15;

        if (totalScore > best.score) {
            best.score = totalScore;
            best.macroGroup = proto.macroGroup;
            best.subGroup = proto.subGroup;
            best.lexicalScore = lexicalScore;
            best.regularityScore = regularityScore;
            best.geographyScore = geographyScore;
        }
    }

    return best;
}

std::string safeGroup(const std::string& s) {
    return s.empty() ? "(unknown)" : s;
}

void printFeaturesSummary(TeeStream& out, const RegionData& region, const PlaceInfo* info) {
    out << "Region: " << region.name;
    if (info != 0) {
        out << " | gold=" << info->macroGroup << "/" << info->subGroup;
    }
    out << "\n";
    out << "  lexical_features=" << region.lexicalFeatures.size()
        << " regularity_features=" << region.regularityFeatures.size() << "\n";
    out << "  regularities:";
    if (region.regularityFeatures.empty()) {
        out << " (none)";
    } else {
        bool first = true;
        for (std::unordered_set<std::string>::const_iterator it = region.regularityFeatures.begin(); it != region.regularityFeatures.end(); ++it) {
            out << (first ? " " : ", ") << *it;
            first = false;
        }
    }
    out << "\n";
}

} // namespace

int main() {
    try {
        setupConsoleUtf8();
        TeeStream out("result.txt", std::cout);

        // 入力データを読み込み、分類に使う基礎データを構築する。
        PlaceLuaOrder placeOrder = parsePlaceLuaOrder("place.lua");
        std::unordered_map<std::string, PlaceInfo> places = parsePlaceLua("place.lua");
        std::unordered_map<std::string, RegionData> regions = parsePlaceTsv("place.tsv");
        CoverageStats coverage = collectCoverageStats("place.tsv");

        // 読み込み件数と、地域ごとの語彙データ充足状況を確認するためのデバッグ出力。
        out << "debug: place.lua keys=" << places.size() << "\n";
        out << "debug: place.tsv regions=" << regions.size() << "\n";
        out << "debug: lexical rows in place.tsv=" << coverage.lexicalRowCount << "\n";

        int matchedPlaces = 0;
        for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
            if (places.find(normalizePlaceKey(it->first)) != places.end()) {
                ++matchedPlaces;
            }
        }
        out << "debug: matched regions=" << matchedPlaces << "\n\n";

        out << "debug: per-region non-empty cell coverage vs extracted lexical features\n";
        std::vector<std::string> regionNames;
        for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
            regionNames.push_back(it->first);
        }
        std::sort(regionNames.begin(), regionNames.end());
        for (std::size_t i = 0; i < regionNames.size(); ++i) {
            const std::string& regionName = regionNames[i];
            const RegionData& region = regions.find(regionName)->second;
            const int nonEmpty = coverage.nonEmptyCellsByRegion.count(regionName) != 0 ? coverage.nonEmptyCellsByRegion[regionName] : -1;
            out << "debug: region=" << regionName
                << " nonEmptyCells=" << nonEmpty
                << " lexicalFeatures=" << region.lexicalFeatures.size()
                << " regularityFeatures=" << region.regularityFeatures.size() << "\n";
        }
        out << "\n";

        // 語彙特徴から追加の規則性特徴を抽出する。
        for (std::unordered_map<std::string, RegionData>::iterator it = regions.begin(); it != regions.end(); ++it) {
            extractRegularityFeatures(it->second);
        }

        // ラベル付き地域だけを学習対象にする。
        std::vector<TrainingSample> samples = buildTrainingSamples(regions, places);
        std::sort(samples.begin(), samples.end(), [](const TrainingSample& a, const TrainingSample& b) {
            return a.name < b.name;
        });

        out << "Ainu dialect bulk classifier\n";
        out << "training_samples=" << samples.size() << "\n\n";

        int macroCorrect = 0;
        int subCorrect = 0;
        std::unordered_map<std::string, RegionSummaryRow> summaryByRegion;

        // 各地域を 1 件ずつ除外して予測する leave-one-out 評価を行う。
        for (std::size_t i = 0; i < samples.size(); ++i) {
            std::vector<Prototype> prototypes = buildPrototypes(samples, samples[i].name);
            ClassificationResult predicted = classifySample(samples[i], prototypes);

            const bool macroOk = predicted.macroGroup == samples[i].macroGroup;
            const bool subOk = macroOk && predicted.subGroup == samples[i].subGroup;
            if (macroOk) {
                ++macroCorrect;
            }
            if (subOk) {
                ++subCorrect;
            }

            RegionSummaryRow row;
            row.region = samples[i].name;
            row.goldMacroGroup = samples[i].macroGroup;
            row.goldSubGroup = samples[i].subGroup;
            row.predictedMacroGroup = predicted.macroGroup;
            row.predictedSubGroup = predicted.subGroup;
            row.lat = samples[i].lat;
            row.lon = samples[i].lon;
            row.nonEmptyCells = coverage.nonEmptyCellsByRegion.count(samples[i].name) != 0 ? coverage.nonEmptyCellsByRegion[samples[i].name] : 0;
            row.lexicalFeatureCount = static_cast<int>(samples[i].lexicalFeatures.size());
            row.regularityFeatureCount = static_cast<int>(samples[i].regularityFeatures.size());
            row.score = predicted.score;
            row.lexicalScore = predicted.lexicalScore;
            row.regularityScore = predicted.regularityScore;
            row.geographyScore = predicted.geographyScore;
            row.matchKind = subOk ? "sub" : (macroOk ? "macro" : "no");
            summaryByRegion[row.region] = row;

            out << samples[i].name
                << "\tgold=" << safeGroup(samples[i].macroGroup) << "/" << safeGroup(samples[i].subGroup)
                << "\tpred=" << safeGroup(predicted.macroGroup) << "/" << safeGroup(predicted.subGroup)
                << "\tmatch=" << (subOk ? "sub" : (macroOk ? "macro" : "no"))
                << "\tscore=" << std::fixed << std::setprecision(3) << predicted.score
                << "\tlex=" << predicted.lexicalScore
                << "\treg=" << predicted.regularityScore
                << "\tgeo=" << predicted.geographyScore
                << "\n";
        }

        out << "\nMacro accuracy: " << macroCorrect << "/" << samples.size()
            << " = " << std::fixed << std::setprecision(3)
            << (samples.empty() ? 0.0 : static_cast<double>(macroCorrect) / static_cast<double>(samples.size()))
            << "\n";
        out << "Subgroup accuracy: " << subCorrect << "/" << samples.size()
            << " = " << std::fixed << std::setprecision(3)
            << (samples.empty() ? 0.0 : static_cast<double>(subCorrect) / static_cast<double>(samples.size()))
            << "\n\n";

        // 学習済み地域について、抽出された特徴量の要約を出力する。
        out << "Feature summaries for labeled regions:\n";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            std::unordered_map<std::string, RegionData>::const_iterator rit = regions.find(samples[i].name);
            std::unordered_map<std::string, PlaceInfo>::const_iterator pit = places.find(samples[i].name);
            if (rit != regions.end()) {
                printFeaturesSummary(out, rit->second, pit != places.end() ? &pit->second : 0);
            }
        }

        // ラベルのない地域は、全学習サンプルから作ったプロトタイプで推定する。
        out << "\nUnlabeled regions in TSV:\n";
        std::vector<Prototype> allPrototypes = buildPrototypes(samples, "");
        for (std::unordered_map<std::string, RegionData>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
            if (places.find(it->first) != places.end()) {
                continue;
            }

            TrainingSample pseudo;
            pseudo.name = it->first;
            pseudo.lat = it->second.lat;
            pseudo.lon = it->second.lon;
            pseudo.lexicalFeatures = it->second.lexicalFeatures;
            pseudo.regularityFeatures = it->second.regularityFeatures;

            ClassificationResult predicted = classifySample(pseudo, allPrototypes);
            RegionSummaryRow row;
            row.region = it->first;
            row.predictedMacroGroup = predicted.macroGroup;
            row.predictedSubGroup = predicted.subGroup;
            row.lat = it->second.lat;
            row.lon = it->second.lon;
            row.nonEmptyCells = coverage.nonEmptyCellsByRegion.count(it->first) != 0 ? coverage.nonEmptyCellsByRegion[it->first] : 0;
            row.lexicalFeatureCount = static_cast<int>(it->second.lexicalFeatures.size());
            row.regularityFeatureCount = static_cast<int>(it->second.regularityFeatures.size());
            row.score = predicted.score;
            row.lexicalScore = predicted.lexicalScore;
            row.regularityScore = predicted.regularityScore;
            row.geographyScore = predicted.geographyScore;
            row.matchKind = "unlabeled";
            summaryByRegion[row.region] = row;

            out << it->first
                << "\tpred=" << safeGroup(predicted.macroGroup) << "/" << safeGroup(predicted.subGroup)
                << "\tscore=" << std::fixed << std::setprecision(3) << predicted.score
                << "\tlex=" << predicted.lexicalScore
                << "\treg=" << predicted.regularityScore
                << "\tgeo=" << predicted.geographyScore
                << "\n";
        }

        std::vector<RegionSummaryRow> summaryRows;
        const std::vector<std::string> orderedNames = collectOrderedRegionNames(regions, placeOrder.names);
        for (std::size_t i = 0; i < orderedNames.size(); ++i) {
            const std::string& regionName = orderedNames[i];
            RegionSummaryRow row;
            if (summaryByRegion.count(regionName) != 0) {
                row = summaryByRegion[regionName];
            } else {
                row.region = regionName;
                row.lat = regions.find(regionName)->second.lat;
                row.lon = regions.find(regionName)->second.lon;
                row.nonEmptyCells = coverage.nonEmptyCellsByRegion.count(regionName) != 0 ? coverage.nonEmptyCellsByRegion[regionName] : 0;
                row.lexicalFeatureCount = static_cast<int>(regions.find(regionName)->second.lexicalFeatures.size());
                row.regularityFeatureCount = static_cast<int>(regions.find(regionName)->second.regularityFeatures.size());
            }
            std::unordered_map<std::string, PlaceInfo>::const_iterator pit = places.find(regionName);
            if (pit != places.end()) {
                row.goldMacroGroup = pit->second.macroGroup;
                row.goldSubGroup = pit->second.subGroup;
            }
            summaryRows.push_back(row);
        }

        const std::unordered_map<std::string, RegionData> matrixRegions =
            filterRegionsByLexicalFeatureCount(regions, 40);

        writeSimilarityMatrixTsv("similarity_matrix.tsv", matrixRegions, placeOrder.names);
        writeDistanceMatrixTsv("distance_matrix.tsv", matrixRegions, placeOrder.names);
        writeRegionSummaryTsv("region_summary.tsv", summaryRows);

        // MDS (多次元尺度法) による 2 次元座標を計算し、散布図用 TSV を出力する。
        {
            const std::vector<std::string> mdsNames = collectOrderedRegionNames(matrixRegions, placeOrder.names);
            const std::size_t n = mdsNames.size();

            // 距離行列 D を構築する。
            std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
            for (std::size_t i = 0; i < n; ++i) {
                const RegionData& li = matrixRegions.find(mdsNames[i])->second;
                for (std::size_t j = i + 1; j < n; ++j) {
                    const RegionData& lj = matrixRegions.find(mdsNames[j])->second;
                    const double d = 1.0 - pairwiseSimilarity(li, lj);
                    D[i][j] = d;
                    D[j][i] = d;
                }
            }

            // Classical MDS: B = -0.5 * H * D^2 * H, where H = I - (1/n)*11^T
            std::vector<std::vector<double>> D2(n, std::vector<double>(n, 0.0));
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    D2[i][j] = D[i][j] * D[i][j];
                }
            }

            // 行平均、列平均、全体平均を計算する。
            std::vector<double> rowMean(n, 0.0);
            std::vector<double> colMean(n, 0.0);
            double grandMean = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    rowMean[i] += D2[i][j];
                    colMean[j] += D2[i][j];
                }
            }
            for (std::size_t i = 0; i < n; ++i) {
                rowMean[i] /= static_cast<double>(n);
                colMean[i] /= static_cast<double>(n);
                grandMean += rowMean[i];
            }
            grandMean /= static_cast<double>(n);

            // B 行列を構築する。
            std::vector<std::vector<double>> B(n, std::vector<double>(n, 0.0));
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    B[i][j] = -0.5 * (D2[i][j] - rowMean[i] - colMean[j] + grandMean);
                }
            }

            // べき乗法で上位 2 固有ベクトルを求める。
            auto matVecMul = [&](const std::vector<std::vector<double>>& M, const std::vector<double>& v) -> std::vector<double> {
                std::vector<double> result(n, 0.0);
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        result[i] += M[i][j] * v[j];
                    }
                }
                return result;
            };

            auto vecNorm = [&](const std::vector<double>& v) -> double {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    s += v[i] * v[i];
                }
                return std::sqrt(s);
            };

            auto vecNormalize = [&](std::vector<double>& v) {
                const double norm = vecNorm(v);
                if (norm > 1e-15) {
                    for (std::size_t i = 0; i < n; ++i) {
                        v[i] /= norm;
                    }
                }
            };

            auto vecDot = [&](const std::vector<double>& a, const std::vector<double>& b) -> double {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    s += a[i] * b[i];
                }
                return s;
            };

            // 第 1 固有ベクトル
            std::vector<double> ev1(n, 1.0);
            vecNormalize(ev1);
            double eigenval1 = 0.0;
            for (int iter = 0; iter < 300; ++iter) {
                std::vector<double> w = matVecMul(B, ev1);
                eigenval1 = vecNorm(w);
                if (eigenval1 < 1e-15) break;
                for (std::size_t i = 0; i < n; ++i) {
                    ev1[i] = w[i] / eigenval1;
                }
            }

            // 第 2 固有ベクトル (デフレーション)
            std::vector<std::vector<double>> B2(n, std::vector<double>(n, 0.0));
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    B2[i][j] = B[i][j] - eigenval1 * ev1[i] * ev1[j];
                }
            }

            std::vector<double> ev2(n);
            for (std::size_t i = 0; i < n; ++i) {
                ev2[i] = (i % 2 == 0) ? 1.0 : -1.0;
            }
            // グラム・シュミットで ev1 と直交化
            {
                double d = vecDot(ev2, ev1);
                for (std::size_t i = 0; i < n; ++i) ev2[i] -= d * ev1[i];
            }
            vecNormalize(ev2);
            double eigenval2 = 0.0;
            for (int iter = 0; iter < 300; ++iter) {
                std::vector<double> w = matVecMul(B2, ev2);
                // 直交化
                double d = vecDot(w, ev1);
                for (std::size_t i = 0; i < n; ++i) w[i] -= d * ev1[i];
                eigenval2 = vecNorm(w);
                if (eigenval2 < 1e-15) break;
                for (std::size_t i = 0; i < n; ++i) {
                    ev2[i] = w[i] / eigenval2;
                }
            }

            // 座標 = sqrt(eigenvalue) * eigenvector
            const double scale1 = eigenval1 > 0.0 ? std::sqrt(eigenval1) : 0.0;
            const double scale2 = eigenval2 > 0.0 ? std::sqrt(eigenval2) : 0.0;

            std::ofstream mdsOut = openUtf8Tsv("mds_scatter.tsv");
            mdsOut << "region\tMDS1\tMDS2\tmacro_group\tsub_group\n";
            for (std::size_t i = 0; i < n; ++i) {
                const std::string& regionName = mdsNames[i];
                std::string macroGroup;
                std::string subGroup;
                std::unordered_map<std::string, PlaceInfo>::const_iterator pit = places.find(regionName);
                if (pit != places.end()) {
                    macroGroup = pit->second.macroGroup;
                    subGroup = pit->second.subGroup;
                }
                if (summaryByRegion.count(regionName) != 0) {
                    const RegionSummaryRow& sr = summaryByRegion[regionName];
                    if (macroGroup.empty()) macroGroup = sr.predictedMacroGroup;
                    if (subGroup.empty()) subGroup = sr.predictedSubGroup;
                }
                mdsOut << regionName << '\t'
                       << formatDouble(scale1 * ev1[i], 6) << '\t'
                       << formatDouble(scale2 * ev2[i], 6) << '\t'
                       << macroGroup << '\t'
                       << subGroup << '\n';
            }
        }

        out << "\nSaved: similarity_matrix.tsv, distance_matrix.tsv, region_summary.tsv, mds_scatter.tsv\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
