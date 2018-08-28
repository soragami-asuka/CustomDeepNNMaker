//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����(�f�o�C�X��������)
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_DEVICE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_DEVICE_H__

#include"FeedforwardNeuralNetwork_GPU_base.cuh"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include<thrust/device_vector.h>

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_GPU_d : public FeedforwardNeuralNetwork_GPU_base
	{
	private:
		std::map<GUID, thrust::device_vector<F32>>	lpLayerOutputBuffer_d;	/**< �e���C���[���Ƃ̏o�̓o�b�t�@(�f�o�C�X������) */

		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_GPU_d();

	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind(void)const;



		//====================================
		// �o�̓o�b�t�@�֘A
		//====================================
	protected:
		/** �o�̓o�b�t�@�̑�����ݒ肷�� */
		ErrorCode SetOutputBufferCount(U32 i_outputBufferCount);

		/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
		ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize);

	public:
		/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
		GUID GetReservedOutputBufferID(U32 i_outputBufferNo);
		/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
		BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid);


	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif