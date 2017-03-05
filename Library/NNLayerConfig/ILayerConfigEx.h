//===========================
// NN�̃��C���[�ݒ荀�ڃt�H�[�}�b�g�x�[�X
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__

#include"NNLayerInterface/ILayerConfig.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerConfigEx : public ILayerConfig
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigEx()
			: ILayerConfig()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigEx(){}

	public:
		/** �A�C�e����ǉ�����.
			�ǉ����ꂽ�A�C�e���͓�����delete�����. */
		virtual int AddItem(ILayerConfigItemBase* pItem)=0;

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_WRITE_ABLE_H__