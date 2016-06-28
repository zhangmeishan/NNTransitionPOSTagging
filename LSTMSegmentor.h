/*
 * Segmentor.h
 *
 *  Created on: Mar 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3L.h"

#include "LSTMBeamSearcher.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Segmentor {
public:
	std::string nullkey;
	std::string rootdepkey;
	std::string unknownkey;
	std::string paddingtag;
	std::string seperateKey;

public:
	Segmentor();
	virtual ~Segmentor();

public:

	LSTMBeamSearcher m_classifier;

	Options m_options;

	Pipe m_pipe;

public:

	int createAlphabet(const vector<Instance>& vecInsts);

	int addTestWordAlpha(const vector<Instance>& vecInsts);

	int allWordAlphaEmb(const string& inFile, NRMat<dtype>& emb);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
			const string& wordEmbFile, const string& charEmbFile, const string& bicharEmbFile, const string& lexiconFile);

	void predict(const Instance& input, CResult& output);

	void test(const string& testFile, const string& outputFile, const string& modelFile, const string& lexiconFile);

	// static training
	void getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);
	public:

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

	void readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& emb);

};

#endif /* SRC_PARSER_H_ */
