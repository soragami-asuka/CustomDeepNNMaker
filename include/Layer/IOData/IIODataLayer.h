//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"../IO/ISingleOutputLayer.h"
#include"../IO/ISingleInputLayer.h"
#include"IDataLayer.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer : public IO::ISingleOutputLayer, public IO::ISingleInputLayer, public IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IIODataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataLayer(){}

	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v */
		virtual ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

		/** �덷�̒l���擾����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			@param	o_min	�ŏ��덷.
			@param	o_max	�ő�덷.
			@param	o_ave	���ό덷.
			@param	o_ave2	���ϓ��덷.
			@param	o_crossEntropy	�N���X�G���g���s�[*/
		virtual ErrorCode GetCalculateErrorValue(F32& o_min, F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy) = 0;

		/** �ڍׂȌ덷�̒l���擾����.
			�e���o�͂̒l���Ɍ덷�����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			�e�z��̗v�f����[GetBufferCount()]�ȏ�ł���K�v������.
			@param	o_lpMin		�ŏ��덷.
			@param	o_lpMax		�ő�덷.
			@param	o_lpAve		���ό덷.
			@param	o_lpAve2	���ϓ��덷. */
		virtual ErrorCode GetCalculateErrorValueDetail(F32 o_lpMin[], F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[]) = 0;
	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif