//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Activation_Discriminator_DATA.hpp"
#include"Activation_Discriminator_FUNC.hpp"
#include"Activation_Discriminator_Base.h"

#include"Activation_Discriminator_GPU.cuh"
#include"Activation_Discriminator_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Activation_Discriminator_GPU::Activation_Discriminator_GPU(Gravisbell::GUID guid, Activation_Discriminator_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Discriminator_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle		(NULL)
		,	inputTensorDesc	(NULL)
		,	outputTensorDesc	(NULL)
	{
		cublasCreate(&cublasHandle);
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateTensorDescriptor(&tmpOutputTensorDesc);
	}
	/** �f�X�g���N�^ */
	Activation_Discriminator_GPU::~Activation_Discriminator_GPU()
	{
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(tmpOutputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
		if(cublasHandle)		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Activation_Discriminator_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Activation_Discriminator_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& Activation_Discriminator_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_Discriminator_GPU::GetLayerData()const
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
	ErrorCode Activation_Discriminator_GPU::PreProcessLearn()
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
	ErrorCode Activation_Discriminator_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �ꎞ�o�̓o�b�t�@���쐬
		this->lpTmpOutputBuffer_d.resize(this->GetBatchSize() * this->inputBufferCount);
		this->lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		{
			int n = this->GetBatchSize();
			int c = this->GetInputDataStruct().ch;
			int h = this->GetInputDataStruct().z * this->GetInputDataStruct().y;
			int w = this->GetInputDataStruct().x;

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

			err = cudnnSetTensorNdDescriptor(this->tmpOutputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

			// ���̓o�b�t�@�������T�C�Y�Ȃ̂Ńf�B�X�N���v�^������Ă���
			err = cudnnSetTensorNdDescriptor(this->inputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// �o�̓o�b�t�@���쐬
		this->lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);	/**< �o�͌덷�o�b�t�@��CPU���A�h���X */
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
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_Discriminator_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Activation_Discriminator_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		F32 alpha = 1.0f;
		F32 beta = 0.0f;
		cudnnStatus_t err =	cudnnSoftmaxForward(
				this->cudnnHandle,
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				&beta,
				this->tmpOutputTensorDesc,
				thrust::raw_pointer_cast(&this->lpTmpOutputBuffer_d[0]));
		if(err != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

		cublasStatus_t err_cublas =	cublasScopy_v2(this->cublasHandle,
			this->GetBatchSize(),
			thrust::raw_pointer_cast(&this->lpTmpOutputBuffer_d[0]),
			this->inputBufferCount,
			o_lppOutputBuffer,
			1);
		if(err_cublas != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;


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
	ErrorCode Activation_Discriminator_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// �o�͌덷���z�X�g���ɃR�s�[
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpDOutputBuffer_h[0]), i_lppDOutputBuffer, sizeof(F32)*this->lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

			// ���͌덷���v�Z
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				//this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 0] = (       this->lpOutputBuffer_d[batchNum]) *  this->lpDOutputBuffer_h[batchNum];
				//this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 1] = (1.0f - this->lpOutputBuffer_d[batchNum]) * -this->lpDOutputBuffer_h[batchNum];
				this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 0] =  this->lpDOutputBuffer_h[batchNum];
				this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 1] = -this->lpDOutputBuffer_h[batchNum];
			}

			cudaMemcpy(o_lppDInputBuffer, thrust::raw_pointer_cast(&this->lpDInputBuffer_h[0]), sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_Discriminator_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
