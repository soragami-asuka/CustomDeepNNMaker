//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// CPU����
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_CPU.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Library/Common/TemporaryMemoryManager.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_CPU::FeedforwardNeuralNetwork_CPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), Common::CreateTemporaryMemoryManagerCPU())
	{
	}
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_CPU::FeedforwardNeuralNetwork_CPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), i_temporaryMemoryManager)
	{
	}

	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_CPU::~FeedforwardNeuralNetwork_CPU()
	{
	}

	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 FeedforwardNeuralNetwork_CPU::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_CPU | Gravisbell::Layer::LAYER_KIND_HOSTMEMORY;
	}


	//====================================
	// ���͌덷�o�b�t�@�֘A
	//====================================
	/** ���͌덷�o�b�t�@�̑�����ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_CPU::SetDInputBufferCount(U32 i_DInputBufferCount)
	{
		this->lpDInputBuffer.resize(i_DInputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���͌덷�o�b�t�@�̃T�C�Y��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_CPU::ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBuffer[i_DInputBufferNo].resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���͌덷�o�b�t�@���擾���� */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::GetDInputBuffer_d(U32 i_DInputBufferNo)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return &this->lpDInputBuffer[i_DInputBufferNo][0];
	}


	//====================================
	// �o�̓o�b�t�@�֘A
	//====================================
	/** �o�̓o�b�t�@�̑�����ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_CPU::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		this->lpOutputBuffer.resize(i_outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_CPU::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpOutputBuffer[i_outputBufferNo].lpBuffer.resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
	GUID FeedforwardNeuralNetwork_CPU::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return GUID();

		return this->lpOutputBuffer[i_outputBufferNo].reserveLayerID;
	}
	/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return NULL;

		this->lpOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;

		return &this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0];
	}
	/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�)
		@param	i_outputBufferNo	�o�̓o�b�t�@�ԍ�
		@param	i_lppBuffer			�o�b�t�@�̏������Ɏg�p����z�X�g�o�b�t�@
		@param	i_bufferSize		�������o�b�t�@�̃T�C�Y. */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid, CONST_BATCH_BUFFER_POINTER i_lppBuffer, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return NULL;

		if(this->lpOutputBuffer[i_outputBufferNo].reserveLayerID != i_guid)
		{
			memcpy(
				&this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0],
				i_lppBuffer,
				sizeof(F32) * min(this->lpOutputBuffer[i_outputBufferNo].lpBuffer.size(), i_bufferSize));

			this->lpOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;
		}

		return &this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0];
	}


	//====================================
	// ���o�̓o�b�t�@�֘A
	//====================================
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, FeedforwardNeuralNetwork_Base::GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	//================================
	// ���Z����
	//================================
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

		// �o�̓o�b�t�@���R�s�[
		if(err == ErrorCode::ERROR_CODE_NONE)
		{
			if(o_lppOutputBuffer)
			{
				memcpy(o_lppOutputBuffer, this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());
			}
		}

		return err;
	}
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@���R�s�[
		if(this->lpInputBuffer.empty())
			this->lpInputBuffer.resize(this->GetInputBufferCount() * this->GetBatchSize());
		memcpy(&this->lpInputBuffer[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer.size());

		return this->Calculate_device(&this->lpInputBuffer[0], NULL);
	}

	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return  CalculateDInput_device(&this->lpInputBuffer[0], o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return  Training_device(&this->lpInputBuffer[0], o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

