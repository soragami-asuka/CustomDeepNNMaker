//======================================
// ��݂��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#include"stdafx.h"

#include"Convolution_LayerData_Base.h"
#include"Convolution_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	Convolution_LayerData_Base::Convolution_LayerData_Base(const Gravisbell::GUID& guid)
		: ISingleInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
		,	inputDataStruct	()	/**< ���̓f�[�^�\�� */
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
	{
	}
	/** �f�X�g���N�^ */
	Convolution_LayerData_Base::~Convolution_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID Convolution_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID Convolution_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode Convolution_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// ���C���[�R�[�h���m�F
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();

		// �\���̂ɓǂݍ���
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* Convolution_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 Convolution_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// ���̓f�[�^�\��
		bufferSize += sizeof(this->inputDataStruct);

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// �{�̂̃o�C�g��
		bufferSize += (this->layerStructure.Output_Channel * this->layerStructure.FilterSize.x * this->layerStructure.FilterSize.y * this->layerStructure.FilterSize.z * this->inputDataStruct.ch) * sizeof(NEURON_TYPE);	// �j���[�����W��
		bufferSize += this->layerStructure.Output_Channel * sizeof(NEURON_TYPE);	// �o�C�A�X�W��


		return bufferSize;
	}


	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct Convolution_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 Convolution_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct Convolution_LayerData_Base::GetOutputDataStruct()const
	{
		IODataStruct outputDataStruct;

		outputDataStruct.x = this->convolutionCountVec.x;
		outputDataStruct.y = this->convolutionCountVec.y;
		outputDataStruct.z = this->convolutionCountVec.z;
		outputDataStruct.ch = this->layerStructure.Output_Channel;

		return outputDataStruct;
	}

	/** �o�̓o�b�t�@�����擾���� */
	unsigned int Convolution_LayerData_Base::GetOutputBufferCount()const
	{
		return this->GetOutputDataStruct().GetDataCount();
	}

	
	//===========================
	// �ŗL�֐�
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
