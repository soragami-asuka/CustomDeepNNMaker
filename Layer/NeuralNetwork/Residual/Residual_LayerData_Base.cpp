//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#include"stdafx.h"

#include"Residual_LayerData_Base.h"
#include"Residual_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Residual_LayerData_Base::Residual_LayerData_Base(const Gravisbell::GUID& guid)
		:	IMultInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
//		,	layerStructure	()		/**< ���C���[�\�� */
	{
	}
	/** �f�X�g���N�^ */
	Residual_LayerData_Base::~Residual_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID Residual_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID Residual_LayerData_Base::GetLayerCode(void)const
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
	ErrorCode Residual_LayerData_Base::Initialize(void)
	{
		// �o�̓f�[�^�\�������肷��

		this->outputDataStruct = this->lpInputDataStruct[0];

		// �������CH����ݒ�
		this->outputDataStruct.ch = 0;
		for(U32 inputNum=0; inputNum<this->lpInputDataStruct.size(); inputNum++)
		{
			this->outputDataStruct.ch = max(this->outputDataStruct.ch, this->lpInputDataStruct[inputNum].ch);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode Residual_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct i_lpInputDataStruct[], U32 i_inputDataCount)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^����ȏ㑶�݂��邱�Ƃ��m�F
		if(i_inputDataCount <= 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// ���̓f�[�^�\���̊e�v�f�������ł��邱�Ƃ��m�F
		for(U32 inputNum=1; inputNum<i_inputDataCount; inputNum++)
		{
			if(i_lpInputDataStruct[inputNum-1].x != i_lpInputDataStruct[inputNum].x)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
			if(i_lpInputDataStruct[inputNum-1].y != i_lpInputDataStruct[inputNum].y)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
			if(i_lpInputDataStruct[inputNum-1].z != i_lpInputDataStruct[inputNum].z)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
		}

		// ���̓f�[�^�\���̐ݒ�
		this->lpInputDataStruct.resize(i_inputDataCount);
		for(U32 inputNum=0; inputNum<i_inputDataCount; inputNum++)
		{
			this->lpInputDataStruct[inputNum] = i_lpInputDataStruct[inputNum];
		}

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode Residual_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// ���̓f�[�^��
		U32 inputDataCount = 0;
		memcpy(&inputDataCount, i_lpBuffer, sizeof(U32));
		readBufferByte += sizeof(U32);
		// ���̓f�[�^�\��
		std::vector<IODataStruct> lpTmpInputDataStruct(inputDataCount);
		for(U32 inputNum=0; inputNum<inputDataCount; inputNum++)
		{
			memcpy(&lpTmpInputDataStruct[inputNum], &i_lpBuffer[readBufferByte], sizeof(IODataStruct));
			readBufferByte += sizeof(IODataStruct);
		}

		// �ݒ���
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// ����������
		ErrorCode err = this->Initialize(*pLayerStructure, &lpTmpInputDataStruct[0], (U32)lpTmpInputDataStruct.size());
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
	ErrorCode Residual_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
//		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* Residual_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 Residual_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// ���̓f�[�^��
		bufferSize += sizeof(U32);

		// ���̓f�[�^�\��
		bufferSize += sizeof(IODataStruct) * (U32)this->lpInputDataStruct.size();

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();


		return bufferSize;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Residual_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// ���̓f�[�^��
		U32 inputDataCount = (U32)this->lpInputDataStruct.size();
		memcpy(&o_lpBuffer[writeBufferByte], &inputDataCount, sizeof(U32));
		writeBufferByte += sizeof(U32);

		// ���̓f�[�^�\��
		for(U32 inputNum=0; inputNum<this->lpInputDataStruct.size(); inputNum++)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpInputDataStruct[inputNum], sizeof(IODataStruct));
			writeBufferByte += sizeof(IODataStruct);
		}

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		return writeBufferByte;
	}



	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�̐����擾���� */
	U32 Residual_LayerData_Base::GetInputDataCount()const
	{
		return (U32)this->lpInputDataStruct.size();
	}

	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct Residual_LayerData_Base::GetInputDataStruct(U32 i_dataNum)const
	{
		if(i_dataNum >= this->lpInputDataStruct.size())
			return IODataStruct(0, 0, 0, 0);

		return this->lpInputDataStruct[i_dataNum];
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 Residual_LayerData_Base::GetInputBufferCount(U32 i_dataNum)const
	{
		return this->GetInputDataStruct(i_dataNum).GetDataCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct Residual_LayerData_Base::GetOutputDataStruct()const
	{
		return this->outputDataStruct;
	}

	/** �o�̓o�b�t�@�����擾���� */
	unsigned int Residual_LayerData_Base::GetOutputBufferCount()const
	{
		return GetOutputDataStruct().GetDataCount();
	}

	
	//===========================
	// �ŗL�֐�
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
