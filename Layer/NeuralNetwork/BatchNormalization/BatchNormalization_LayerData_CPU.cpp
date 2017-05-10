//======================================
// �o�b�`���K���̃��C���[�f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"BatchNormalization_LayerData_CPU.h"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	BatchNormalization_LayerData_CPU::BatchNormalization_LayerData_CPU(const Gravisbell::GUID& guid)
		:	BatchNormalization_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalization_LayerData_CPU::~BatchNormalization_LayerData_CPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_LayerData_CPU::Initialize(void)
	{
		this->lpMean.resize(this->inputDataStruct.ch);
		this->lpVariance.resize(this->inputDataStruct.ch);
		this->lpScale.resize(this->inputDataStruct.ch);
		this->lpBias.resize(this->inputDataStruct.ch);

		for(U32 ch=0; ch<this->inputDataStruct.ch; ch++)
		{
			this->lpMean[ch] = 0.0f;
			this->lpVariance[ch] = 0.0f;
			this->lpScale[ch] = 1.0f;
			this->lpBias[ch] = 0.0f;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^�\���̐ݒ�
		this->inputDataStruct = i_inputDataStruct;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// ���̓f�[�^�\��
		memcpy(&this->inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(this->inputDataStruct));
		readBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		// ����
		memcpy(&this->lpMean[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// ���U
		memcpy(&this->lpVariance[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// �X�P�[�����O�l
		memcpy(&this->lpScale[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// �o�C�A�X�l
		memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());


		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 BatchNormalization_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// ���̓f�[�^�\��
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		// ����
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpMean[0], sizeof(F32)*this->lpMean.size());
		writeBufferByte += sizeof(F32)*this->lpMean.size();
		// ���U
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpVariance[0], sizeof(F32)*this->lpVariance.size());
		writeBufferByte += sizeof(F32)*this->lpVariance.size();
		// �X�P�[�����O�l
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpScale[0], sizeof(F32)*this->lpScale.size());
		writeBufferByte += sizeof(F32)*this->lpScale.size();
		// �o�C�A�X�l
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], sizeof(F32)*this->lpBias.size());
		writeBufferByte += sizeof(F32)*this->lpBias.size();


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	INNLayer* BatchNormalization_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new BatchNormalization_CPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// �ǂݎ��Ɏg�p����o�b�t�@�����擾
	U32 useBufferSize = pLayerData->GetUseBufferByteCount();
	if(useBufferSize >= (U32)i_bufferSize)
	{
		delete pLayerData;
		return NULL;
	}

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
