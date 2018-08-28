//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_h.cuh"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Library/Common/TemporaryMemoryManager.h"

// CUDA�p
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda_runtime_api.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_h::FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount)
	{
	}
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_h::FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount, i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_h::~FeedforwardNeuralNetwork_GPU_h()
	{
	}

	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 FeedforwardNeuralNetwork_GPU_h::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_GPU | Gravisbell::Layer::LAYER_KIND_HOSTMEMORY;
	}


	//====================================
	// �o�̓o�b�t�@�֘A
	//====================================
	/** �o�̓o�b�t�@�̑�����ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_h::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		this->lpTemporaryOutputBuffer.resize(i_outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_h::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer.resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
	GUID FeedforwardNeuralNetwork_GPU_h::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return GUID();

		return this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID;
	}
	/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_h::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return NULL;

		if(this->lpLayerOutputBuffer_h.count(i_guid) == 0)
		{
			// ���C���[�̏o�̓o�b�t�@���m�ۂ���
			IODataStruct outputDataStruct = this->GetOutputDataStruct(i_guid);

			this->lpLayerOutputBuffer_h[i_guid].resize(outputDataStruct.GetDataCount() * this->GetBatchSize());
		}

		// �Ώۂ̃o�b�t�@��\�񒆂łȂ��ꍇ�A���݂̃o�b�t�@���z�X�g���ɑޔ�.�V�����o�b�t�@�̓��e���f�o�C�X���ɃR�s�[
		if(this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID != i_guid)
		{
			// �ޔ�
			{
				auto it_buffer = this->lpLayerOutputBuffer_h.find(this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID);
				if(it_buffer != this->lpLayerOutputBuffer_h.end())
				{
					cudaMemcpy(
						&it_buffer->second[0],
						thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]),
						sizeof(F32) * it_buffer->second.size(),
						cudaMemcpyDeviceToHost);
				}
			}

			// �f�[�^�R�s�[
			{
				auto it_buffer = this->lpLayerOutputBuffer_h.find(i_guid);
				if(it_buffer != this->lpLayerOutputBuffer_h.end())
				{
					cudaMemcpy(
						thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]),
						&it_buffer->second[0],
						sizeof(F32) * it_buffer->second.size(),
						cudaMemcpyHostToDevice);
				}
			}
		}

		// �\������X�V
		this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;

		return thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]);
	}



}	// NeuralNetwork
}	// Layer
}	// Gravisbell

