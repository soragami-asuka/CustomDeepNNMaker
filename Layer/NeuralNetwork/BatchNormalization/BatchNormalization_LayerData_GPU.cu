//======================================
// �o�b�`���K���̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"BatchNormalization_LayerData_GPU.cuh"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_GPU.cuh"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	BatchNormalization_LayerData_GPU::BatchNormalization_LayerData_GPU(const Gravisbell::GUID& guid)
		:	BatchNormalization_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalization_LayerData_GPU::~BatchNormalization_LayerData_GPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_LayerData_GPU::Initialize(void)
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
	ErrorCode BatchNormalization_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
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
	ErrorCode BatchNormalization_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize )
	{
		int readBufferByte = 0;

		// ���̓f�[�^�\��
		memcpy(&this->inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(this->inputDataStruct));
		readBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		// ����
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpMean[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpMean.size();
		// ���U
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpVariance[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpMean.size();
		// �X�P�[�����O�l
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpScale[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpMean.size();
		// �o�C�A�X�l
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpMean.size();


		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 BatchNormalization_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
		cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpMean[0]), sizeof(F32)*this->lpMean.size(), cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32)*(U32)this->lpMean.size();
		// ���U
		cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpVariance[0]), sizeof(F32)*this->lpVariance.size(), cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32)*(U32)this->lpVariance.size();
		// �X�P�[�����O�l
		cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpScale[0]), sizeof(F32)*this->lpScale.size(), cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32)*(U32)this->lpScale.size();
		// �o�C�A�X�l
		cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpBias[0]), sizeof(F32)*this->lpBias.size(), cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32)*(U32)this->lpBias.size();

		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* BatchNormalization_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new BatchNormalization_GPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	S32 useBufferSize = 0;
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize, useBufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
