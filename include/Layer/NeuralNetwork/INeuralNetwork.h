//=======================================
// �j���[�����l�b�g���[�N�{�̒�`
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNSingleInputLayer.h"
#include"INNSingleOutputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNSingleInputLayer, public INNSingleOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INeuralNetwork(){}
		/** �f�X�g���N�^ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// �w�K�ݒ�
		//====================================
		/** �w�K�ݒ���擾����.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID. */
		virtual const SettingData::Standard::IData* GetLearnSettingData(const Gravisbell::GUID& guid)const = 0;

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const wchar_t* i_dataID, S32 i_param) = 0;
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const wchar_t* i_dataID, F32 i_param) = 0;
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			bool�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const wchar_t* i_dataID, bool i_param) = 0;
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			string�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetLearnSettingData(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
