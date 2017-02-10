//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __I_INPUT_DATA_LAYER_H__
#define __I_INPUT_DATA_LAYER_H__

#include"ISingleOutputLayer.h"
#include"ISingleInputLayer.h"
#include"IDataLayer.h"

namespace CustomDeepNNLibrary
{
	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer : public ISingleOutputLayer, public ISingleInputLayer, public IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IIODataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataLayer(){}

	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			���O�̌v�Z���ʂ��g�p���� */
		virtual ELayerErrorCode CalculateLearnError(const float** i_lppInputBuffer) = 0;

	};
}

#endif