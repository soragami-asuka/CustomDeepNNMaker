//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#include"stdafx.h"

#include"FullyConnect_LayerData_Base.h"
#include"FullyConnect_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	FullyConnect_LayerData_Base::FullyConnect_LayerData_Base(const Gravisbell::GUID& guid)
		:	ISingleInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
		,	inputDataStruct	()	/**< ���̓f�[�^�\�� */
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
		,	m_pOptimizer_neuron	(NULL)		/**< �j���[�����X�V�p�I�v�e�B�}�C�U */
		,	m_pOptimizer_bias	(NULL)		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */
	{
	}
	/** �f�X�g���N�^ */
	FullyConnect_LayerData_Base::~FullyConnect_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;

		if(this->m_pOptimizer_neuron)
			delete this->m_pOptimizer_neuron;
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID FullyConnect_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID FullyConnect_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode FullyConnect_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* FullyConnect_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 FullyConnect_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// ���̓f�[�^�\��
		bufferSize += sizeof(this->inputDataStruct);

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// �{�̂̃o�C�g��
		bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// �j���[�����W��
		bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// �o�C�A�X�W��

		// �I�v�e�B�}�C�U�[�̃o�C�g��
		bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();
		bufferSize += this->m_pOptimizer_neuron->GetUseBufferByteCount();


		return bufferSize;
	}


	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct FullyConnect_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 FullyConnect_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct FullyConnect_LayerData_Base::GetOutputDataStruct()const
	{
		IODataStruct outputDataStruct;

		outputDataStruct.x = 1;
		outputDataStruct.y = 1;
		outputDataStruct.z = 1;
		outputDataStruct.ch = this->GetNeuronCount();

		return outputDataStruct;
	}

	/** �o�̓o�b�t�@�����擾���� */
	unsigned int FullyConnect_LayerData_Base::GetOutputBufferCount()const
	{
		IODataStruct outputDataStruct = GetOutputDataStruct();

		return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
	}

	
	//===========================
	// �ŗL�֐�
	//===========================
	/** �j���[���������擾���� */
	U32 FullyConnect_LayerData_Base::GetNeuronCount()const
	{
		return this->layerStructure.NeuronCount;
	}


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================
	/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
	ErrorCode FullyConnect_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FullyConnect_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);
		
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FullyConnect_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
