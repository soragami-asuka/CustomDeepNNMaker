//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_H__

#include"FeedforwardNeuralNetwork_Base.h"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include<thrust/device_vector.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_GPU : public FeedforwardNeuralNetwork_Base
	{
	private:
		std::vector<thrust::device_vector<F32>> lpDInputBuffer;

		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct);
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_GPU();

	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind(void)const;


		//====================================
		// ���͌덷�o�b�t�@�֘A
		//====================================
	protected:
		/** ���͌덷�o�b�t�@�̑�����ݒ肷�� */
		ErrorCode SetDInputBufferCount(U32 i_DInputBufferCount);

		/** ���͌덷�o�b�t�@�̃T�C�Y��ݒ肷�� */
		ErrorCode ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize);

	public:
		/** ���͌덷�o�b�t�@���擾���� */
		BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_DInputBufferNo);
		/** ���͌덷�o�b�t�@���擾���� */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_DInputBufferNo)const;


	public:
		//====================================
		// ���o�̓o�b�t�@�֘A
		//====================================
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

		/** �w�K�������擾����.
			@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif