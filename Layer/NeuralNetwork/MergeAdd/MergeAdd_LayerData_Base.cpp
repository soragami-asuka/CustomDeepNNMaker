//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#include"stdafx.h"

#include"MergeAdd_LayerData_Base.h"
#include"MergeAdd_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	MergeAdd_LayerData_Base::MergeAdd_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
	{
	}
	/** �f�X�g���N�^ */
	MergeAdd_LayerData_Base::~MergeAdd_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID MergeAdd_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID MergeAdd_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeAdd_LayerData_Base::Initialize(void)
	{
		// �o�̓f�[�^�\�������肷��


		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode MergeAdd_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode MergeAdd_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// �ݒ���
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// ����������
		ErrorCode err = this->Initialize(*pLayerStructure);
		delete pLayerStructure;
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode MergeAdd_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* MergeAdd_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 MergeAdd_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();


		return bufferSize;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 MergeAdd_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		return writeBufferByte;
	}


	
	//===========================
	// ���C���[�\��
	//===========================
	/** ���̓f�[�^�\�����g�p�\���m�F����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
	bool MergeAdd_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		
		// ���̓f�[�^�\���̊e�v�f�������ł��邱�Ƃ��m�F
		for(U32 inputNum=1; inputNum<i_inputLayerCount; inputNum++)
		{
			if(i_lpInputDataStruct[inputNum-1].x != i_lpInputDataStruct[inputNum].x)
			{
				return false;
			}
			if(i_lpInputDataStruct[inputNum-1].y != i_lpInputDataStruct[inputNum].y)
			{
				return false;
			}
			if(i_lpInputDataStruct[inputNum-1].z != i_lpInputDataStruct[inputNum].z)
			{
				return false;
			}
		}

		return true;
	}

	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct MergeAdd_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);
		
		IODataStruct outputDataStruct = i_lpInputDataStruct[0];

		// �������CH����ݒ�
		outputDataStruct.ch = i_lpInputDataStruct[0].ch;
		switch(this->layerStructure.MergeType)
		{
		case MergeAdd::LayerStructure::MergeType_max:
			for(U32 inputNum=0; inputNum<i_inputLayerCount; inputNum++)
			{
				outputDataStruct.ch = max(outputDataStruct.ch, i_lpInputDataStruct[inputNum].ch);
			}
			break;
		case MergeAdd::LayerStructure::MergeType_min:
			for(U32 inputNum=0; inputNum<i_inputLayerCount; inputNum++)
			{
				outputDataStruct.ch = min(outputDataStruct.ch, i_lpInputDataStruct[inputNum].ch);
			}
			break;
		case MergeAdd::LayerStructure::MergeType_layer0:
			outputDataStruct.ch = i_lpInputDataStruct[0].ch;
			break;
		}

		return outputDataStruct;
	}

	/** �����o�͂��\�����m�F���� */
	bool MergeAdd_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}

	
	//===========================
	// �ŗL�֐�
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
