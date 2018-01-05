//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_d.cuh"

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
	FeedforwardNeuralNetwork_GPU_d::FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_inputDataStruct)
	{
	}
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_d::FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_d::~FeedforwardNeuralNetwork_GPU_d()
	{
	}

	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 FeedforwardNeuralNetwork_GPU_d::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_GPU | Gravisbell::Layer::LAYER_KIND_DEVICEMEMORY;
	}


	//====================================
	// �o�̓o�b�t�@�֘A
	//====================================
	/** �o�̓o�b�t�@�̑�����ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_d::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_d::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
	GUID FeedforwardNeuralNetwork_GPU_d::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		return GUID();
	}
	/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_d::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(this->lpLayerOutputBuffer_d.count(i_guid) == 0)
		{
			// ���C���[�̏o�̓o�b�t�@���m�ۂ���
			IODataStruct outputDataStruct = this->GetOutputDataStruct(i_guid);

			this->lpLayerOutputBuffer_d[i_guid].resize(outputDataStruct.GetDataCount() * this->GetBatchSize());
		}

		return thrust::raw_pointer_cast(&this->lpLayerOutputBuffer_d[i_guid][0]);
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

