/*
 * CTagConstraints.h
 *
 *  Created on: Apr 23, 2016
 *      Author: panda
 */

#ifndef BASIC_TAG_H_
#define BASIC_TAG_H_

#include <hash_set>
#include "Hash_map.hpp"
#include "Alphabet.h"
#include "Utf.h"

class CTagConstraints {

public:

	// from training
	hash_map<string, hash_set<string> > word_pos;
	hash_map<string, int> word_freq;
	int maxFreqWord;

	//later we will add
	// cluster
	// character muter-information
	// word muter-information
	// word-pos muter-information
	// word: low freq, high-freq

public:

	CTagConstraints() {
		clear();
	}

	inline void clear() {
		word_pos.clear();
		word_freq.clear();
		maxFreqWord = 0;

	}

public:

	inline void addWordPOSPair(const string& word, const string& postag) {
		static vector<string> charInfo;
		static string prefix;

		word_pos[word].insert(postag);
		word_freq[word]++;
		if (word_freq[word] > maxFreqWord) {
			maxFreqWord = word_freq[word];
		}

	}

	inline bool inWord_posMap(const string& theWord, const string& thePostag) {

		if (word_freq.find(theWord) != word_freq.end() && word_pos.find(theWord) != word_pos.end()
				&& word_freq[theWord] > (maxFreqWord / 5000 + 3)) {
			if (word_pos[theWord].find(thePostag) != word_pos[theWord].end()) {
				return false;
			}
		}
		return false;
	}

	void write(std::ofstream &outf) {
		hash_map<string, hash_set<string> >::iterator hashmapIt;
		hash_set<string>::iterator hashsetIt;
		hash_map<string, int>::iterator it;

		outf << 1 << endl << maxFreqWord << endl;

		outf << word_pos.size() << endl;
		for (hashmapIt = word_pos.begin(); hashmapIt != word_pos.end(); hashmapIt++) {
			outf << hashmapIt->first << " " << hashmapIt->second.size() << endl;
			for (hashsetIt = hashmapIt->second.begin(); hashsetIt != hashmapIt->second.end(); hashsetIt++) {
				outf << *hashsetIt << endl;
			}
			outf << endl;
		}

		outf << word_freq.size() << endl;
		for (it = word_freq.begin(); it != word_freq.end(); it++) {
			outf << it->first << " " << it->second << endl;
		}

	}

	void read(std::ifstream &inf) {
		clear();
		static string tmp;
		static vector<string> modelInfo;
		static int size;

		my_getline(inf, tmp);
		chomp(tmp);
		size = atoi(tmp.c_str());
		for (int i = 0; i < size; ++i) {
			my_getline(inf, tmp);
			split_bychars(tmp, modelInfo);
			maxFreqWord = atoi(modelInfo[0].c_str());
		}

		my_getline(inf, tmp);
		chomp(tmp);
		size = atoi(tmp.c_str());

		for (int i = 0; i < size; ++i) {
			my_getline(inf, tmp);
			split_bychars(tmp, modelInfo);
			int count = atoi(modelInfo[1].c_str());
			for (int j = 0; j < count; ++j) {
				my_getline(inf, tmp);
				word_pos[modelInfo[0]].insert(tmp);
			}
		}

		my_getline(inf, tmp);
		chomp(tmp);
		size = atoi(tmp.c_str());
		for (int i = 0; i < size; ++i) {
			my_getline(inf, tmp);
			split_bychars(tmp, modelInfo);
			word_freq[modelInfo[0]] = atoi(modelInfo[1].c_str());
		}

	}

	inline void showStatics() {
		cout << "word number: " << word_pos.size() << endl;

		cout << "word max frequency in training data: " << maxFreqWord << endl;

	}

	void readExternalInformation(const string& inFile, int cutOff = 0) {
		static ifstream inf;
		if (inf.is_open()) {
			inf.close();
			inf.clear();
		}
		inf.open(inFile.c_str());

		static string strLine, curWord, curPos, prefix;
		static int i, count, sumFreq, curFreq;
		static vector<string> vecInfo, charInfo;

		while (1) {
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (!strLine.empty()) {
				split_bychar(strLine, vecInfo, ' ');
				count = vecInfo.size();
				if (count % 2 == 0) {
					std::cout << "error line :" << strLine << std::endl;
					continue;
				}

				curWord = vecInfo[0];

				getCharactersFromUTF8String(curWord, charInfo);

				sumFreq = 0;
				for (i = 1; i < count; i = i + 2) {
					curPos = vecInfo[i];
					curFreq = atoi(vecInfo[i + 1].c_str());
					sumFreq += curFreq;
				}

				if (sumFreq > cutOff) {
					for (i = 1; i < count; i = i + 2) {
						curPos = vecInfo[i];

					}
				}

			}
		}
	}


};

#endif /* BASIC_TAG_H_ */
