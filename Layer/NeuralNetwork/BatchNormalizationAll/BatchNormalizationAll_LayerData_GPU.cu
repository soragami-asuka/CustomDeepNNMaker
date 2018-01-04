//======================================
// �o�b�`���K���̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"BatchNormalizationAll_LayerData_GPU.cuh"
#include"BatchNormalizationAll_FUNC.hpp"
#include"BatchNormalizationAll_GPU.cuh"

#include"Library/NeuralNetwork/Optimizer.h"

#include"../_LayerBase/CLayerBase_GPU.cuh"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	BatchNormalizationAll_LayerData_GPU::BatchNormalizationAll_LayerData_GPU(const Gravisbell::GUID& guid)
		:	BatchNormalizationAll_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalizationAll_LayerData_GPU::~BatchNormalizationAll_LayerData_GPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalizationAll_LayerData_GPU::Initialize(void)
	{
		this->lpMean.resize(1);
		this->lpVariance.resize(1);
		this->lpScale.resize(1);
		this->lpBias.resize(1);

		for(U32 ch=0; ch<1; ch++)
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
	ErrorCode BatchNormalizationAll_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ������
		err = this->Initialize();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// �I�v�e�B�}�C�U�[�̐ݒ�
		err = this->ChangeOptimizer(L"SGD");
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode BatchNormalizationAll_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize )
	{
		int readBufferByte = 0;

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
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpVariance[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpVariance.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpVariance.size();
		// �X�P�[�����O�l
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpScale[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpScale.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpScale.size();
		// �o�C�A�X�l
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]), &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpBias.size(), cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32)*(U32)this->lpBias.size();


		// �I�v�e�B�}�C�U
		S32 useBufferSize = 0;
		// bias
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
		this->m_pOptimizer_bias = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
		readBufferByte += useBufferSize;
		// neuron
		if(this->m_pOptimizer_scale)
			delete this->m_pOptimizer_scale;
		this->m_pOptimizer_scale = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
		readBufferByte += useBufferSize;


		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 BatchNormalizationAll_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

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


		// �I�v�e�B�}�C�U
		// bias
		writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
		// neuron
		writeBufferByte += this->m_pOptimizer_scale->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* BatchNormalizationAll_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_GPU<BatchNormalizationAll_GPU, BatchNormalizationAll_LayerData_GPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}
	

	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================		
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode BatchNormalizationAll_LayerData_GPU::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		ChangeOptimizer_GPU(&this->m_pOptimizer_bias,  i_optimizerID, (U32)this->lpBias.size());
		ChangeOptimizer_GPU(&this->m_pOptimizer_scale, i_optimizerID, (U32)this->lpScale.size());

		return ErrorCode::ERROR_CODE_NONE;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::BatchNormalizationAll_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalizationAll_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data);
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
	Gravisbell::Layer::NeuralNetwork::BatchNormalizationAll_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalizationAll_LayerData_GPU(guid);
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
