#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "N3L.h"
#include "Metric.h"
#include "Alphabet.h"
#include "CResult.h"

using namespace std;

class Instance {
public:
	Instance() {
	}
	~Instance() {
	}

	int wordsize() const {
		return words.size();
	}

	int charsize() const {
		return chars.size();
	}

	int postagsize() const {
		return postags.size();
	}


	void clear() {
		words.clear();
		chars.clear();
		postags.clear();
		for(int i = 0; i < candidateLabels.size(); i++) {
			candidateLabels[i].clear();
		}
		candidateLabels.clear();
	}

	void allocate(int length, int charLength) {
		clear();
		words.resize(length);
		chars.resize(charLength);
		postags.resize(length);
		candidateLabels.resize(charLength);
	}

	void copyValuesFrom(const Instance& anInstance) {
		allocate(anInstance.wordsize(), anInstance.charsize());
		for (int i = 0; i < anInstance.wordsize(); i++) {
			words[i] = anInstance.words[i];
			postags[i] = anInstance.postags[i];
		}
		for (int i = 0; i < anInstance.charsize(); i++) {
			chars[i] = anInstance.chars[i];
			candidateLabels[i].clear();
			for(int j = 0; j < anInstance.candidateLabels[i].size(); j++){
				candidateLabels[i].push_back(anInstance.candidateLabels[i][j]);
			}
		}
	}

	void evaluate(const CResult& curResult, Metric& segEval, Metric& tagEval) const {
		hash_set<string> seggolds, segpreds;
		getSegIndexes(words, seggolds);
		getSegIndexes(curResult.words, segpreds);

		hash_set<string> taggolds, tagpreds;
		getPosIndexes(words, postags, taggolds);
		getPosIndexes(curResult.words, curResult.postags, tagpreds);

		hash_set<string>::iterator iter;
		segEval.overall_label_count += seggolds.size();
		segEval.predicated_label_count += segpreds.size();
		for (iter = segpreds.begin(); iter != segpreds.end(); iter++) {
			if (seggolds.find(*iter) != seggolds.end()) {
				segEval.correct_label_count++;
			}
		}

		tagEval.overall_label_count += taggolds.size();
		tagEval.predicated_label_count += tagpreds.size();
		for (iter = tagpreds.begin(); iter != tagpreds.end(); iter++) {
			if (taggolds.find(*iter) != taggolds.end()) {
				tagEval.correct_label_count++;
			}
		}
	}

	void getSegIndexes(const vector<string>& segs, hash_set<string>& segIndexes) const{
	  segIndexes.clear();
	  int idx = 0, idy = 0;
	  string curWord = "";
	  int beginId = 0;
	  while(idx < chars.size() && idy < segs.size()){
	    curWord = curWord + chars[idx];
	    if(curWord.length() == segs[idy].length()){
        stringstream ss;
        ss << "[" << beginId << "," << idx << "]";
        segIndexes.insert(ss.str());
        idy++;
        beginId = idx+1;
        curWord = "";
	    }
	    idx++;
	  }

	  if(idx != chars.size() || idy != segs.size()){
	    std::cout << "error segs, please check" << std::endl;
	  }
	}

	void getPosIndexes(const vector<string>& segs, const vector<string>& postags, hash_set<string>& postagIndexes) const {
		postagIndexes.clear();

	  int idx = 0, idy = 0;
	  string curWord = "";
	  int beginId = 0;
	  while(idx < chars.size() && idy < segs.size()){
	    curWord = curWord + chars[idx];
	    if(curWord.length() == segs[idy].length()){
        stringstream ss;
        ss << "[" << beginId << "," << idx << "]" << postags[idy];
        postagIndexes.insert(ss.str());
        idy++;
        beginId = idx+1;
        curWord = "";
	    }
	    idx++;
	  }

	  if(idx != chars.size() || idy != segs.size()){
	    std::cout << "error segs, please check" << std::endl;
	  }
	}


public:
	vector<string> words;
	vector<string> chars;
	vector<string> postags;
	vector<vector<string> > candidateLabels;
};

#endif

