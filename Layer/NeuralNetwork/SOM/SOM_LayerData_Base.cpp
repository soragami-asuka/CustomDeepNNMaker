//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#include"stdafx.h"

#include"SOM_LayerData_Base.h"
#include"SOM_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	SOM_LayerData_Base::SOM_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
	{
	}
	/** �f�X�g���N�^ */
	SOM_LayerData_Base::~SOM_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;

	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID SOM_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID SOM_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode SOM_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* SOM_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U64 SOM_LayerData_Base::GetUseBufferByteCount()const
	{
		U64 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;


		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// �{�̂̃o�C�g��
		bufferSize += (this->GetUnitCount() * this->layerStructure.InputBufferCount) * sizeof(F32);	// �j���[�����W��
		bufferSize += this->GetUnitCount() * sizeof(F32);	// �o�C�A�X�W��

		return bufferSize;
	}


	//===========================
	// ���C���[�\��
	//===========================
	/** ���̓f�[�^�\�����g�p�\���m�F����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
	bool SOM_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		if(i_lpInputDataStruct[0].GetDataCount() != this->layerStructure.InputBufferCount)
			return false;

		return true;
	}

	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct SOM_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);

		return IODataStruct(this->layerStructure.DimensionCount, 1, 1, 1);
	}

	/** �����o�͂��\�����m�F���� */
	bool SOM_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	
	//===========================
	// �ŗL�֐�
	//===========================
	/** ���̓o�b�t�@�����擾���� */
	U32 SOM_LayerData_Base::GetInputBufferCount()const
	{
		return this->layerStructure.InputBufferCount;
	}
	/** ���j�b�g�����擾���� */
	U32 SOM_LayerData_Base::GetUnitCount()const
	{
		return (U32)pow(this->layerStructure.ResolutionCount, this->layerStructure.DimensionCount);
	}


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode SOM_LayerData_Base::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		return ERROR_CODE_NONE;
	}
	/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode SOM_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//==================================
	// SOM�֘A����
	//==================================
	/** �}�b�v�T�C�Y���擾����.
		@return	�}�b�v�̃o�b�t�@����Ԃ�. */
	U32 SOM_LayerData_Base::GetMapSize()const
	{
		return this->GetUnitCount() * this->GetInputBufferCount();
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
