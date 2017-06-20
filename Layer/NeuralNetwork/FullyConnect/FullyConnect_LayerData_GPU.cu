//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"FullyConnect_LayerData_GPU.cuh"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_GPU.cuh"

#include"RandomUtility.h"

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"Library/NeuralNetwork/Optimizer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	FullyConnect_LayerData_GPU::FullyConnect_LayerData_GPU(const Gravisbell::GUID& guid)
		:	FullyConnect_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	FullyConnect_LayerData_GPU::~FullyConnect_LayerData_GPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_LayerData_GPU::Initialize(void)
	{
		// �����Œ艻
		Utility::Random::Initialize(0);

		// ���̓o�b�t�@�����m�F
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		this->lppNeuron_d.reserve(neuronCount * inputBufferCount);
		this->lppNeuron_d.resize(neuronCount * inputBufferCount);
		this->lpBias_d.reserve(neuronCount);
		this->lpBias_d.resize(neuronCount);

		thrust::host_vector<F32> lpTmpNeuron(neuronCount * inputBufferCount);
		thrust::host_vector<F32> lpTmpBias(neuronCount);

		float maxArea = sqrt(6.0f / (inputBufferCount + neuronCount));
		for(U32 i=0; i<lpTmpNeuron.size(); i++)
			lpTmpNeuron[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;
		for(U32 i=0; i<lpTmpBias.size(); i++)
			lpTmpBias[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;

		this->lppNeuron_d = lpTmpNeuron;
		this->lpBias_d = lpTmpBias;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^�\���̐ݒ�
		this->inputDataStruct = i_inputDataStruct;

		// ������
		err = this->Initialize();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// �I�v�e�B�}�C�U�[�̐ݒ�
		err = this->ChangeOptimizer(L"SGD");
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize )
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

		// �o�b�t�@����R�s�[
		// �j���[����
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lppNeuron_d[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lppNeuron_d.size(),
			cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32) * (S32)this->lppNeuron_d.size();

		// �o�C�A�X
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lpBias_d[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lpBias_d.size(),
			cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32) * (S32)this->lpBias_d.size();


		// �I�v�e�B�}�C�U
		S32 useBufferSize = 0;
		// bias
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
		this->m_pOptimizer_bias = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
		readBufferByte += useBufferSize;
		// neuron
		if(this->m_pOptimizer_neuron)
			delete this->m_pOptimizer_neuron;
		this->m_pOptimizer_neuron = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
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
	S32 FullyConnect_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// ���̓f�[�^�\��
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// �j���[����
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lppNeuron_d[0]),
			sizeof(F32) * this->lppNeuron_d.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lppNeuron_d.size();

		// �o�C�A�X
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lpBias_d[0]),
			sizeof(F32) * this->lpBias_d.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lpBias_d.size();


		// �I�v�e�B�}�C�U
		// bias
		writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
		// neuron
		writeBufferByte += this->m_pOptimizer_neuron->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* FullyConnect_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new FullyConnect_GPU(guid, *this);
	}


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode FullyConnect_LayerData_GPU::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		ChangeOptimizer_GPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias_d.size());
		ChangeOptimizer_GPU(&this->m_pOptimizer_neuron, i_optimizerID, (U32)this->lppNeuron_d.size());

		return ErrorCode::ERROR_CODE_NONE;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU(guid);
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
	Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU(guid);
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