//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#include"stdafx.h"

#include"GaussianNoise_LayerData_Base.h"
#include"GaussianNoise_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	GaussianNoise_LayerData_Base::GaussianNoise_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		,	layerStructure	()		/**< ���C���[�\�� */
	{
	}
	/** �f�X�g���N�^ */
	GaussianNoise_LayerData_Base::~GaussianNoise_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// ���ʏ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID GaussianNoise_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID GaussianNoise_LayerData_Base::GetLayerCode(void)const
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
	ErrorCode GaussianNoise_LayerData_Base::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode GaussianNoise_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
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
	ErrorCode GaussianNoise_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
	{
		S64 readBufferByte = 0;

		// �ݒ���
		S64 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode GaussianNoise_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* GaussianNoise_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U64 GaussianNoise_LayerData_Base::GetUseBufferByteCount()const
	{
		U64 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();


		return bufferSize;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S64 GaussianNoise_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

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
	bool GaussianNoise_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;
		if(i_lpInputDataStruct[0].x == 0)
			return false;
		if(i_lpInputDataStruct[0].y == 0)
			return false;
		if(i_lpInputDataStruct[0].z == 0)
			return false;
		if(i_lpInputDataStruct[0].ch == 0)
			return false;

		return true;
	}

	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct GaussianNoise_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);

		return i_lpInputDataStruct[0];
	}

	/** �����o�͂��\�����m�F���� */
	bool GaussianNoise_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	//===========================
	// �ŗL�֐�
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
