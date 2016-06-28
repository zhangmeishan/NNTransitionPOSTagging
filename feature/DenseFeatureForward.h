/*
 * DenseFeatureForward.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATUREFORWARD_H_
#define FEATURE_DENSEFEATUREFORWARD_H_

#include "N3L.h"
class DenseFeatureForward {
public:
	//state inter dependent features
	//word
	Tensor<cpu, 3, dtype> _wordPrime;
	Tensor<cpu, 3, dtype> _allwordPrime;
	Tensor<cpu, 3, dtype> _keyCharPrime;
	Tensor<cpu, 3, dtype> _lengthPrime;
	Tensor<cpu, 2, dtype> _wordRep, _allwordRep, _keyCharRep, _lengthRep;
	Tensor<cpu, 2, dtype> _wordUnitRep, _wordUnitRepMask;
	Tensor<cpu, 2, dtype> _wordHidden;
	vector<Tensor<cpu, 2, dtype> > _wordRNNHiddenBuf;
	Tensor<cpu, 2, dtype> _wordRNNHidden;  //lstm

	//postag
	Tensor<cpu, 3, dtype> _postagPrime;
	Tensor<cpu, 2, dtype> _postagRep, _postagRepMask;
	Tensor<cpu, 2, dtype> _postagHidden;
	vector<Tensor<cpu, 2, dtype> > _postagRNNHiddenBuf;
	Tensor<cpu, 2, dtype> _postagRNNHidden;  //lstm

	//action
	Tensor<cpu, 3, dtype> _actionPrime;
	Tensor<cpu, 2, dtype> _actionRep, _actionRepMask;
	Tensor<cpu, 2, dtype> _actionHidden;
	vector<Tensor<cpu, 2, dtype> > _actionRNNHiddenBuf;
	Tensor<cpu, 2, dtype> _actionRNNHidden;  //lstm

	//state inter independent features
	Tensor<cpu, 2, dtype> _sepInHidden, _sepInHiddenLoss;  //sep in
	Tensor<cpu, 2, dtype> _sepOutHidden, _sepOutHiddenLoss;  //sep out
	Tensor<cpu, 2, dtype> _appInHidden, _appInHiddenLoss;  //app in
	Tensor<cpu, 2, dtype> _appOutHidden, _appOutHiddenLoss;  //app out

	bool _bAllocated;
	bool _bTrain;
	int _buffer;

	int _wordDim, _allwordDim, _lengthDim, _charDim, _wordNgram, _wordHiddenDim, _wordRNNDim, _wordUnitDim;
	int _postagDim, _postagNgram, _postagPreDim, _postagHiddenDim, _postagRNNDim;
	int _actionDim, _actionNgram, _actionPreDim, _actionHiddenDim, _actionRNNDim;
	int _sepInHiddenDim, _appInHiddenDim, _sepOutHiddenDim, _appOutHiddenDim;

public:
	DenseFeatureForward() {
		_bAllocated = false;
		_bTrain = false;
		_buffer = 0;

		_wordDim = 0;
		_allwordDim = 0;
		_charDim = 0;
		_lengthDim = 0;
		_wordNgram = 0;
		_wordUnitDim = 0;
		_wordHiddenDim = 0;
		_wordRNNDim = 0;

		_postagDim = 0;
		_postagNgram = 0;
		_postagPreDim = 0;
		_postagHiddenDim = 0;
		_postagRNNDim = 0;

		_actionDim = 0;
		_actionNgram = 0;
		_actionPreDim = 0;
		_actionHiddenDim = 0;
		_actionRNNDim = 0;

		_sepInHiddenDim = 0;
		_appInHiddenDim = 0;
		_sepOutHiddenDim = 0;
		_appOutHiddenDim = 0;
	}

	~DenseFeatureForward() {
		clear();
	}

public:
	inline void init(int wordDim, int allwordDim, int charDim, int lengthDim, int wordNgram, int wordHiddenDim, int wordRNNDim,
			int postagDim, int postagNgram, int postagHiddenDim, int postagRNNDim,
			int actionDim, int actionNgram, int actionHiddenDim, int actionRNNDim,
			int sepInHiddenDim, int appInHiddenDim, int sepOutHiddenDim, int appOutHiddenDim, int buffer = 0, bool bTrain = false) {
		clear();
		_buffer = buffer;

		_wordDim = wordDim;
		_allwordDim = allwordDim;
		_charDim = charDim;
		_lengthDim = lengthDim;
		_wordNgram = wordNgram;
		_wordUnitDim = wordNgram * wordDim + wordNgram * allwordDim + (2 * wordNgram + 1) * charDim + wordNgram * lengthDim;
		_wordHiddenDim = wordHiddenDim;
		_wordRNNDim = wordRNNDim;

		_wordPrime = NewTensor<cpu>(Shape3(_wordNgram, 1, _wordDim), d_zero);
		_allwordPrime = NewTensor<cpu>(Shape3(_wordNgram, 1, _allwordDim), d_zero);
		_wordRep = NewTensor<cpu>(Shape2(1, _wordNgram * _wordDim), d_zero);
		_allwordRep = NewTensor<cpu>(Shape2(1, _wordNgram * _allwordDim), d_zero);
		_keyCharPrime = NewTensor<cpu>(Shape3(2 * _wordNgram + 1, 1, _charDim), d_zero);
		_keyCharRep = NewTensor<cpu>(Shape2(1, (2 * _wordNgram + 1) * _charDim), d_zero);
		_lengthPrime = NewTensor<cpu>(Shape3(_wordNgram, 1, _lengthDim), d_zero);
		_lengthRep = NewTensor<cpu>(Shape2(1, _wordNgram * _lengthDim), d_zero);
		_wordUnitRep = NewTensor<cpu>(Shape2(1, _wordUnitDim), d_zero);
		_wordHidden = NewTensor<cpu>(Shape2(1, _wordHiddenDim), d_zero);
		if (_buffer > 0) {
			_wordRNNHiddenBuf.resize(_buffer);
			for (int idk = 0; idk < _buffer; idk++) {
				_wordRNNHiddenBuf[idk] = NewTensor<cpu>(Shape2(1, _wordRNNDim), d_zero);
			}
		}
		_wordRNNHidden = NewTensor<cpu>(Shape2(1, _wordRNNDim), d_zero);

		_postagDim = postagDim;
		_postagNgram = postagNgram;
		_postagPreDim = postagNgram * postagDim;
		_postagHiddenDim = postagHiddenDim;
		_postagRNNDim = postagRNNDim;
		_postagPrime = NewTensor<cpu>(Shape3(_postagNgram, 1, _postagDim), d_zero);
		_postagRep = NewTensor<cpu>(Shape2(1, _postagPreDim), d_zero);
		_postagHidden = NewTensor<cpu>(Shape2(1, _postagHiddenDim), d_zero);
		if (_buffer > 0) {
			_postagRNNHiddenBuf.resize(_buffer);
			for (int idk = 0; idk < _buffer; idk++) {
				_postagRNNHiddenBuf[idk] = NewTensor<cpu>(Shape2(1, _postagRNNDim), d_zero);
			}
		}
		_postagRNNHidden = NewTensor<cpu>(Shape2(1, _postagRNNDim), d_zero);

		_actionDim = actionDim;
		_actionNgram = actionNgram;
		_actionPreDim = actionNgram * actionDim;
		_actionHiddenDim = actionHiddenDim;
		_actionRNNDim = actionRNNDim;
		_actionPrime = NewTensor<cpu>(Shape3(_actionNgram, 1, _actionDim), d_zero);
		_actionRep = NewTensor<cpu>(Shape2(1, _actionPreDim), d_zero);
		_actionHidden = NewTensor<cpu>(Shape2(1, _actionHiddenDim), d_zero);
		if (_buffer > 0) {
			_actionRNNHiddenBuf.resize(_buffer);
			for (int idk = 0; idk < _buffer; idk++) {
				_actionRNNHiddenBuf[idk] = NewTensor<cpu>(Shape2(1, _actionRNNDim), d_zero);
			}
		}
		_actionRNNHidden = NewTensor<cpu>(Shape2(1, _actionRNNDim), d_zero);

		_sepInHiddenDim = sepInHiddenDim;
		_appInHiddenDim = appInHiddenDim;
		_sepOutHiddenDim = sepOutHiddenDim;
		_appOutHiddenDim = appOutHiddenDim;
		_sepInHidden = NewTensor<cpu>(Shape2(1, _sepInHiddenDim), d_zero);
		_appInHidden = NewTensor<cpu>(Shape2(1, _appInHiddenDim), d_zero);
		_sepOutHidden = NewTensor<cpu>(Shape2(1, _sepOutHiddenDim), d_zero);
		_appOutHidden = NewTensor<cpu>(Shape2(1, _appOutHiddenDim), d_zero);

		if (bTrain) {
			_wordUnitRepMask = NewTensor<cpu>(Shape2(1, _wordUnitDim), d_zero);
			_actionRepMask = NewTensor<cpu>(Shape2(1, _actionPreDim), d_zero);
			_postagRepMask = NewTensor<cpu>(Shape2(1, _postagPreDim), d_zero);

			_sepInHiddenLoss = NewTensor<cpu>(Shape2(1, _sepInHiddenDim), d_zero);
			_appInHiddenLoss = NewTensor<cpu>(Shape2(1, _appInHiddenDim), d_zero);
			_sepOutHiddenLoss = NewTensor<cpu>(Shape2(1, _sepOutHiddenDim), d_zero);
			_appOutHiddenLoss = NewTensor<cpu>(Shape2(1, _appOutHiddenDim), d_zero);

		}

		_bTrain = bTrain;
		_bAllocated = true;
	}

	inline void clear() {
		if (_bAllocated) {
			_wordDim = 0;
			_allwordDim = 0;
			_charDim = 0;
			_lengthDim = 0;
			_wordNgram = 0;
			_wordUnitDim = 0;
			_wordHiddenDim = 0;
			_wordRNNDim = 0;

			FreeSpace(&_wordPrime);
			FreeSpace(&_allwordPrime);
			FreeSpace(&_wordRep);
			FreeSpace(&_allwordRep);
			FreeSpace(&_keyCharPrime);
			FreeSpace(&_keyCharRep);
			FreeSpace(&_lengthPrime);
			FreeSpace(&_lengthRep);
			FreeSpace(&_wordUnitRep);
			FreeSpace(&_wordHidden);
			FreeSpace(&_wordRNNHidden);
			if (_buffer > 0) {
				for (int idk = 0; idk < _buffer; idk++) {
					FreeSpace(&(_wordRNNHiddenBuf[idk]));
				}
				_wordRNNHiddenBuf.clear();
			}

			_postagDim = 0;
			_postagNgram = 0;
			_postagPreDim = 0;
			_postagHiddenDim = 0;
			_postagRNNDim = 0;
			FreeSpace(&_postagPrime);
			FreeSpace(&_postagRep);
			FreeSpace(&_postagHidden);
			FreeSpace(&_postagRNNHidden);
			if (_buffer > 0) {
				for (int idk = 0; idk < _buffer; idk++) {
					FreeSpace(&(_postagRNNHiddenBuf[idk]));
				}
				_postagRNNHiddenBuf.clear();
			}

			_actionDim = 0;
			_actionNgram = 0;
			_actionPreDim = 0;
			_actionHiddenDim = 0;
			_actionRNNDim = 0;
			FreeSpace(&_actionPrime);
			FreeSpace(&_actionRep);
			FreeSpace(&_actionHidden);
			FreeSpace(&_actionRNNHidden);
			if (_buffer > 0) {
				for (int idk = 0; idk < _buffer; idk++) {
					FreeSpace(&(_actionRNNHiddenBuf[idk]));
				}
				_actionRNNHiddenBuf.clear();
			}

			_sepInHiddenDim = 0;
			_appInHiddenDim = 0;
			_sepOutHiddenDim = 0;
			_appOutHiddenDim = 0;
			FreeSpace(&_sepInHidden);
			FreeSpace(&_appInHidden);
			FreeSpace(&_sepOutHidden);
			FreeSpace(&_appOutHidden);

			if (_bTrain) {
				FreeSpace(&_wordUnitRepMask);
				FreeSpace(&_actionRepMask);
				FreeSpace(&_postagRepMask);

				FreeSpace(&_sepInHiddenLoss);
				FreeSpace(&_appInHiddenLoss);
				FreeSpace(&_sepOutHiddenLoss);
				FreeSpace(&_appOutHiddenLoss);
			}

			_bAllocated = false;
			_bTrain = false;
			_buffer = 0;
		}
	}

	inline void copy(const DenseFeatureForward& other) {
		if (other._bAllocated) {
			if (_bAllocated && _bTrain == other._bTrain) {
				if (_wordDim != other._wordDim || _allwordDim != other._allwordDim || _charDim != other._charDim || _lengthDim != other._lengthDim
						|| _wordNgram != other._wordNgram || _wordHiddenDim != other._wordHiddenDim || _wordRNNDim != other._wordRNNDim
						|| _postagDim != other._postagDim || _postagNgram != other._postagNgram || _postagHiddenDim != other._postagHiddenDim || _postagRNNDim != other._postagRNNDim
						|| _actionDim != other._actionDim || _actionNgram != other._actionNgram || _actionHiddenDim != other._actionHiddenDim || _actionRNNDim != other._actionRNNDim
						|| _sepInHiddenDim != other._sepInHiddenDim || _appInHiddenDim != other._appInHiddenDim
						|| _sepOutHiddenDim != other._sepOutHiddenDim || _appOutHiddenDim != other._appOutHiddenDim
						|| _buffer != other._buffer) {
					std::cout << "please check, error allocatation somewhere" << std::endl;
					return;
				}
			} else {
				init(other._wordDim, other._allwordDim, other._charDim, other._lengthDim, other._wordNgram, other._wordHiddenDim, other._wordRNNDim,
						other._postagDim, other._postagNgram, other._postagHiddenDim, other._postagRNNDim,
						other._actionDim, other._actionNgram, other._actionHiddenDim, other._actionRNNDim,
						other._sepInHiddenDim, other._appInHiddenDim, other._sepOutHiddenDim, other._appOutHiddenDim, other._buffer, other._bTrain);
			}
			Copy(_wordPrime, other._wordPrime);
			Copy(_allwordPrime, other._allwordPrime);
			Copy(_wordRep, other._wordRep);
			Copy(_allwordRep, other._allwordRep);
			Copy(_keyCharPrime, other._keyCharPrime);
			Copy(_keyCharRep, other._keyCharRep);
			Copy(_lengthPrime, other._lengthPrime);
			Copy(_lengthRep, other._lengthRep);
			Copy(_wordUnitRep, other._wordUnitRep);
			Copy(_wordHidden, other._wordHidden);
			for (int idk = 0; idk < _buffer; idk++) {
				Copy(_wordRNNHiddenBuf[idk], other._wordRNNHiddenBuf[idk]);
			}
			Copy(_wordRNNHidden, other._wordRNNHidden);

			Copy(_postagPrime, other._postagPrime);
			Copy(_postagRep, other._postagRep);
			Copy(_postagHidden, other._postagHidden);
			for (int idk = 0; idk < _buffer; idk++) {
				Copy(_postagRNNHiddenBuf[idk], other._postagRNNHiddenBuf[idk]);
			}
			Copy(_postagRNNHidden, other._postagRNNHidden);

			Copy(_actionPrime, other._actionPrime);
			Copy(_actionRep, other._actionRep);
			Copy(_actionHidden, other._actionHidden);
			for (int idk = 0; idk < _buffer; idk++) {
				Copy(_actionRNNHiddenBuf[idk], other._actionRNNHiddenBuf[idk]);
			}
			Copy(_actionRNNHidden, other._actionRNNHidden);

			Copy(_sepInHidden, other._sepInHidden);
			Copy(_appInHidden, other._appInHidden);
			Copy(_sepOutHidden, other._sepOutHidden);
			Copy(_appOutHidden, other._appOutHidden);

			if (other._bTrain) {
				Copy(_wordUnitRepMask, other._wordUnitRepMask);
				Copy(_actionRepMask, other._actionRepMask);
				Copy(_postagRepMask, other._postagRepMask);

				Copy(_sepInHiddenLoss, other._sepInHiddenLoss);
				Copy(_appInHiddenLoss, other._appInHiddenLoss);
				Copy(_sepOutHiddenLoss, other._sepOutHiddenLoss);
				Copy(_appOutHiddenLoss, other._appOutHiddenLoss);
			}
		} else {
			clear();
		}

		_bAllocated = other._bAllocated;
		_bTrain = other._bTrain;
		_buffer = other._buffer;
	}

};

#endif /* FEATURE_DENSEFEATUREFORWARD_H_ */
