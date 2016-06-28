/*
 * CResult.h
 *
 *  Created on: May 10, 2016
 *      Author: masonzms
 */

#ifndef BASIC_CRESULT_H_
#define BASIC_CRESULT_H_

using namespace std;

class CResult {
public:
	CResult() {
	}
	~CResult() {
	}

public:
	vector<string> words;
	vector<string> postags;


public:
	void clear(){
		words.clear();
		postags.clear();
	}

	void resize(int size){
		words.resize(size);
		postags.resize(size);
	}

	int size() const{
		return words.size();
	}
};



#endif /* BASIC_CRESULTS_H_ */
