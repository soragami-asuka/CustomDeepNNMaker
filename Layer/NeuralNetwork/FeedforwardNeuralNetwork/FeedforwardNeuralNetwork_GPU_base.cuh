//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����(�f�o�C�X��������)
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_BASE_H__

#include"FeedforwardNeuralNetwork_Base.h"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include<thrust/device_vector.h>

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_GPU_base : public FeedforwardNeuralNetwork_Base
	{
	private:
		std::vector<F32> lpOutputBuffer_h;	/**< �o�̓o�b�t�@(�z�X�g������) */
		std::vector<thrust::device_vector<F32>> lpDInputBuffer;

		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct);
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_GPU_base();


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
		BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_DInputBufferNo);
		/** ���͌덷�o�b�t�@���擾���� */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_DInputBufferNo)const;


		//====================================
		// �o�̓o�b�t�@�֘A
		//====================================
	protected:
		/** �o�̓o�b�t�@�̑�����ݒ肷�� */
		virtual ErrorCode SetOutputBufferCount(U32 i_outputBufferCount) = 0;

		/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
		virtual ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
		virtual GUID GetReservedOutputBufferID(U32 i_outputBufferNo) = 0;
		/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
		virtual BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid) = 0;


	public:
		//====================================
		// ���o�̓o�b�t�@�֘A
		//====================================
		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

	public:
		//================================
		// ���Z����
		//================================
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLearn(U32 batchSize);
		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);


		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer);

		//================================
		// �w�K����
		//================================
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif