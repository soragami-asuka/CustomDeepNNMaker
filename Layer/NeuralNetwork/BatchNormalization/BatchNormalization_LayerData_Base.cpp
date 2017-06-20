//======================================
// �������֐����C���[�̃f�[�^
//======================================
#include"stdafx.h"

#include"BatchNormalization_LayerData_Base.h"
#include"BatchNormalization_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	BatchNormalization_LayerData_Base::BatchNormalization_LayerData_Base(const Gravisbell::GUID& guid)
		:	ISingleInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
		,	inputDataStruct	()	/**< ���̓f�[�^�\�� */
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
		,	m_pOptimizer_scale	(NULL)		/**< �X�P�[���X�V�p�I�v�e�B�}�C�U */
		,	m_pOptimizer_bias	(NULL)		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalization_LayerData_Base::~BatchNormalization_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;

		if(this->m_pOptimizer_scale)
			delete this->m_pOptimizer_scale;
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID BatchNormalization_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID BatchNormalization_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}



	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode BatchNormalization_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* BatchNormalization_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 BatchNormalization_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// ���̓f�[�^�\��
		bufferSize += sizeof(this->inputDataStruct);

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// �e�f�[�^��
		bufferSize += sizeof(F32) * this->inputDataStruct.ch;	// ����
		bufferSize += sizeof(F32) * this->inputDataStruct.ch;	// ���U
		bufferSize += sizeof(F32) * this->inputDataStruct.ch;	// �X�P�[�����O�l
		bufferSize += sizeof(F32) * this->inputDataStruct.ch;	// �o�C�A�X�l

		// �I�v�e�B�}�C�U�[�̃o�C�g��
		bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();
		bufferSize += this->m_pOptimizer_scale->GetUseBufferByteCount();

		return bufferSize;
	}



	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct BatchNormalization_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 BatchNormalization_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct BatchNormalization_LayerData_Base::GetOutputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** �o�̓o�b�t�@�����擾���� */
	unsigned int BatchNormalization_LayerData_Base::GetOutputBufferCount()const
	{
		IODataStruct outputDataStruct = GetOutputDataStruct();

		return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
	}

	
	//===========================
	// �ŗL�֐�
	//===========================


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================
	/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
	ErrorCode BatchNormalization_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_scale)
			this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode BatchNormalization_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_scale)
			this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
		
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode BatchNormalization_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_scale)
			this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
