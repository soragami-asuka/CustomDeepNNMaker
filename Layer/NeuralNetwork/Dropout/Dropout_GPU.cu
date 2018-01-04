//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Dropout_DATA.hpp"
#include"Dropout_FUNC.hpp"
#include"Dropout_Base.h"

#include"Dropout_GPU.cuh"
#include"Dropout_LayerData_GPU.cuh"

#include<time.h>

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Dropout_GPU::Dropout_GPU(Gravisbell::GUID guid, Dropout_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Dropout_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	cudnnHandle		(NULL)
		,	dropoutDesc		(NULL)
		,	inputTensorDesc	(NULL)
		,	outputTensorDesc	(NULL)
		,	m_pState		(NULL)
		,	m_pReserve		(NULL)
		,	reserveSize		(0)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateDropoutDescriptor(&dropoutDesc);
	}
	/** �f�X�g���N�^ */
	Dropout_GPU::~Dropout_GPU()
	{
		if(this->m_pState)		cudaFree(this->m_pState);
		if(this->m_pReserve)	cudaFree(this->m_pReserve);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(dropoutDesc)			cudnnDestroyDropoutDescriptor(dropoutDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Dropout_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Dropout_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Dropout_LayerData_Base& Dropout_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Dropout_LayerData_Base& Dropout_GPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �o�̓o�b�t�@���쐬
		{
			int n = this->GetBatchSize();
			int c = this->GetOutputDataStruct().ch;
			int h = this->GetOutputDataStruct().z * this->GetOutputDataStruct().y;
			int w = this->GetOutputDataStruct().x;

			const int nDims = 4;
			int dimA[nDims] = {n, c, h, w};
			int strideA[nDims] = {c*h*w, h*w, w, 1};

			cudnnStatus_t err = cudnnSetTensorNdDescriptor(this->outputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

			err = cudnnSetTensorNdDescriptor(this->inputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// �h���b�v�A�E�g�ݒ���쐬
		{
			// �����W�F�l���[�^�p���������m��
			size_t stateSize = 0;
			{
				if(this->m_pState != NULL)
				{
					cudaFree(this->m_pState);
					this->m_pState = NULL;
				}

				cudnnDropoutGetStatesSize(this->cudnnHandle, &stateSize);

				cudaError err = cudaMalloc((void**)&this->m_pState, stateSize);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;
			}
			
			// �h���b�v�A�E�g�o�b�t�@�p���������m��
			{
				if(this->m_pReserve != NULL)
				{
					cudaFree(this->m_pReserve);
					this->m_pReserve = NULL;
				}

				cudnnStatus_t cudnnErr = cudnnDropoutGetReserveSpaceSize(this->inputTensorDesc, &this->reserveSize);
				if(cudnnErr != 0)
					return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

				cudaError cudaErr = cudaMalloc((void**)&this->m_pReserve, this->reserveSize);
				if(cudaErr != 0)
					return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;
			}

			// �h���b�v�A�E�g�ݒ���쐬
			{
				cudnnStatus_t err = cudnnSetDropoutDescriptor(
					this->dropoutDesc,
					this->cudnnHandle,
					this->layerData.layerStructure.Rate,
					this->m_pState,
					stateSize,
					(U64)time(NULL));

				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Dropout_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetRuntimeParameterByStructure().UseDropOut)
		{
			cudnnStatus_t err = cudnnDropoutForward(
				this->cudnnHandle,
				this->dropoutDesc,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->outputTensorDesc,
				o_lppOutputBuffer,
				this->m_pReserve,
				this->reserveSize);

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}
		else
		{
			cudaMemcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}



	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Dropout_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			if(this->GetRuntimeParameterByStructure().UseDropOut)
			{
				cudnnStatus_t err = cudnnDropoutBackward(
					this->cudnnHandle,
					this->dropoutDesc,
					this->outputTensorDesc,
					i_lppDOutputBuffer,
					this->inputTensorDesc,
					o_lppDInputBuffer,
					this->m_pReserve,
					this->reserveSize);

				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}
			else
			{
				cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Dropout_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}




} // Gravisbell;
} // Layer;
} // NeuralNetwork;
