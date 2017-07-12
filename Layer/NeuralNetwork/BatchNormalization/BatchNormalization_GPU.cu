//======================================
// �o�b�`���K�����C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"BatchNormalization_DATA.hpp"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_Base.h"

#include"BatchNormalization_GPU.cuh"
#include"BatchNormalization_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	BatchNormalization_GPU::BatchNormalization_GPU(Gravisbell::GUID guid, BatchNormalization_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	BatchNormalization_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount		(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount		(0)				/**< �o�̓o�b�t�@�� */
		,	channeclBufferCount		(0)				/**< 1�`�����l��������̃o�b�t�@�� */
		,	onLearnMode				(false)			/**< �w�K�������t���O */
		,	learnCount				(0)				/**< �w�K���s�� */
		,	m_lppInputBuffer				(NULL)			/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
	{
        cudnnCreate(&this->cudnnHandle);
		cudnnCreateTensorDescriptor(&this->paramTensorDesc);
        cudnnCreateTensorDescriptor(&this->inputTensorDesc);
        cudnnCreateTensorDescriptor(&this->outputTensorDesc);
	}
	/** �f�X�g���N�^ */
	BatchNormalization_GPU::~BatchNormalization_GPU()
	{
        cudnnDestroyTensorDescriptor(this->inputTensorDesc);
        cudnnDestroyTensorDescriptor(this->outputTensorDesc);
		cudnnDestroyTensorDescriptor(this->paramTensorDesc);
        cudnnDestroy(this->cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 BatchNormalization_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	BatchNormalization_LayerData_Base& BatchNormalization_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const BatchNormalization_LayerData_Base& BatchNormalization_GPU::GetLayerData()const
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
	ErrorCode BatchNormalization_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�p�̕ϐ����쐬
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->GetInputDataStruct().ch, 0.0f);
		this->lpTmpVariance.resize(this->GetInputDataStruct().ch, 0.0f);

		// �p�����[�^�ω���
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDScale.resize(this->layerData.lpScale.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_GPU::PreProcessCalculate()
	{
		cudnnStatus_t err_cudnn;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �`�����l�����Ƃ̃o�b�t�@�����m�F
		this->channeclBufferCount = this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);


		// �������𒲂ׂ�
		S32 dataDim = 1 + 1 + 0;	// �o�b�` + �`�����l�� + ����0
		std::vector<S32> dimInput;			// ���̓f�[�^�\��
		std::vector<S32> dimInputStride;	// ���̓f�[�^�̊e�������Ƃ̃f�[�^��
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		std::vector<S32> dimParam;
		std::vector<S32> dimParamStride;
		if(this->GetInputDataStruct().z > 1)
		{
			dataDim = 1 + 1 + 3;	// �o�b�` + �`�����l�� + ����3

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().z;
			dimInput[3] = this->GetInputDataStruct().y;
			dimInput[4] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().z;
			dimOutput[3] = this->GetOutputDataStruct().y;
			dimOutput[4] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[2] = dimOutput[3] * dimOutput[4];
			dimOutputStride[3] = dimOutput[4];
			dimOutputStride[4] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;
			dimParam[3] = 1;
			dimParam[4] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2] * dimParam[3] * dimParam[4];
			dimParamStride[1] = dimParam[2] * dimParam[3] * dimParam[4];
			dimParamStride[2] = dimParam[3] * dimParam[4];
			dimParamStride[3] = dimParam[4];
			dimParamStride[4] = 1;
		}
		else if(this->GetInputDataStruct().y > 1 || this->GetInputDataStruct().x)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().y;
			dimInput[3] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().y;
			dimOutput[3] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;
			dimParam[3] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2] * dimParam[3];
			dimParamStride[1] = dimParam[2] * dimParam[3];
			dimParamStride[2] = dimParam[3];
			dimParamStride[3] = 1;
		}
		else if(this->GetInputDataStruct().x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2];
			dimOutputStride[1] = dimOutput[2];
			dimOutputStride[2] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2];
			dimParamStride[1] = dimParam[2];
			dimParamStride[2] = 1;
		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}


		// CUDNN�̓��̓f�[�^�\�����쐬
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̏o�̓f�[�^�\�����쐬
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̃p�����[�^�f�[�^�\�����쐬
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->paramTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimParam[0],
			&dimParamStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_GPU::PreProcessLoop()
	{
		switch(this->GetProcessType())
		{
		case ProcessType::PROCESSTYPE_LEARN:
			{
				// �w�K�񐔂�������
				this->learnCount = 0;

				// ���Z�p�̕���.���U��������
				cudaMemset(thrust::raw_pointer_cast(&this->layerData.lpMean[0]),	 0, sizeof(F32)*this->GetInputDataStruct().ch);
				cudaMemset(thrust::raw_pointer_cast(&this->layerData.lpVariance[0]), 0, sizeof(F32)*this->GetInputDataStruct().ch);

				cudaMemset(thrust::raw_pointer_cast(&this->lpLearnMean[0]),		0, sizeof(F32)*this->GetInputDataStruct().ch);
				cudaMemset(thrust::raw_pointer_cast(&this->lpLearnVariance[0]),	0, sizeof(F32)*this->GetInputDataStruct().ch);
			}
			break;
		case ProcessType::PROCESSTYPE_CALCULATE:
			{
				// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
				this->lpTmpMean = this->layerData.lpMean;
				this->lpTmpVariance = this->layerData.lpVariance;

				this->lpLearnMean = this->layerData.lpMean;
				this->lpLearnVariance = this->layerData.lpVariance;
			}
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode BatchNormalization_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���̓o�b�t�@�̃A�h���X���i�[
		this->m_lppInputBuffer = i_lpInputBuffer;

		// �w�K���Ȃ�Ε��ρA���U�����߂�
		if(this->onLearnMode)
		{
			// �w�K���̏ꍇ
			F32 alpha = 1.0f;
			F32 beta = 0.0f;

			std::vector<F32> lpVarianceLast(this->GetInputDataStruct().ch);
			for(U32 i=0; i<lpVarianceLast.size(); i++)
				lpVarianceLast[i] = this->layerData.lpVariance[i];

			// ���ρA���U���w�K�p�Ɉڂ�
			this->lpLearnMean     = this->layerData.lpMean;
			this->lpLearnVariance = this->layerData.lpVariance;

			err_cudnn = cudnnBatchNormalizationForwardTraining(
				this->cudnnHandle,
				cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				this->inputTensorDesc,
				this->m_lppInputBuffer,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				this->paramTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
				thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
				(1.0 / (this->learnCount+1)),
				thrust::raw_pointer_cast(&this->lpLearnMean[0]),
				thrust::raw_pointer_cast(&this->lpLearnVariance[0]),
				max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
				thrust::raw_pointer_cast(&this->lpTmpMean[0]),
				thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}
		else
		{
			// �w�K���łȂ��ꍇ
			F32 alpha = 1.0f;
			F32 beta = 0.0f;

			err_cudnn = cudnnBatchNormalizationForwardInference(
				this->cudnnHandle,
				cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				this->inputTensorDesc,
				this->m_lppInputBuffer,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				this->paramTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
				thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
				thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
				thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
				max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode BatchNormalization_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32) * outputBufferCount * batchSize, cudaMemcpyDeviceToHost);

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
	ErrorCode BatchNormalization_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// ���͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(this->m_lpDInputBuffer_d == NULL)
		{
			// ���͌덷�o�b�t�@�����݂��Ȃ��ꍇ�w�K���ł��Ȃ����߁A��փo�b�t�@���m��
			if(this->m_lpTemporaryDInputBuffer_d.size() != this->inputBufferCount * this->GetBatchSize())
				this->m_lpTemporaryDInputBuffer_d.resize(this->inputBufferCount * this->GetBatchSize());

			this->m_lpDInputBuffer_d = thrust::raw_pointer_cast(&this->m_lpTemporaryDInputBuffer_d[0]);
		}


		F32 alphaData = 1.0f;
		F32 betaData  = 0.0f;

		F32 alphaParam = 0.0f;
		F32 betaParam  = 1.0f;

		err_cudnn = cudnnBatchNormalizationBackward(
			this->cudnnHandle,
			cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
			&alphaData,
			&betaData,
			&alphaParam,
			&betaParam,
			this->inputTensorDesc,
			this->m_lppInputBuffer,
			this->outputTensorDesc,
			this->m_lppDOutputBufferPrev,
			this->inputTensorDesc,
			this->m_lpDInputBuffer_d,
			this->paramTensorDesc,
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
			max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
			thrust::raw_pointer_cast(&this->lpTmpMean[0]),
			thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], this->m_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode BatchNormalization_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// ���͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(this->m_lpDInputBuffer_d == NULL)
		{
			// ���͌덷�o�b�t�@�����݂��Ȃ��ꍇ�w�K���ł��Ȃ����߁A��փo�b�t�@���m��
			if(this->m_lpTemporaryDInputBuffer_d.size() != this->inputBufferCount * this->GetBatchSize())
				this->m_lpTemporaryDInputBuffer_d.resize(this->inputBufferCount * this->GetBatchSize());

			this->m_lpDInputBuffer_d = thrust::raw_pointer_cast(&this->m_lpTemporaryDInputBuffer_d[0]);
		}


		F32 alphaData = 1.0f;
		F32 betaData  = 0.0f;

		F32 alphaParam = 1.0F;
		F32 betaParam  = 0.0F;

		err_cudnn = cudnnBatchNormalizationBackward(
			this->cudnnHandle,
			cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
			&alphaData,
			&betaData,
			&alphaParam,
			&betaParam,
			this->inputTensorDesc,
			this->m_lppInputBuffer,
			this->outputTensorDesc,
			this->m_lppDOutputBufferPrev,
			this->inputTensorDesc,
			this->m_lpDInputBuffer_d,
			this->paramTensorDesc,
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->lpDScale[0]),
			thrust::raw_pointer_cast(&this->lpDBias[0]),
			max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
			thrust::raw_pointer_cast(&this->lpTmpMean[0]),
			thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

		// ���ρA���U���X�V
		this->layerData.lpMean = this->lpLearnMean;
		this->layerData.lpVariance = this->lpLearnVariance;

		// �p�����[�^���X�V
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpScale[0]), thrust::raw_pointer_cast(&this->lpDScale[0]));
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias[0]), thrust::raw_pointer_cast(&this->lpDBias[0]));

		// �w�K�����̎��s�񐔂��J�E���g�A�b�v
		this->learnCount++;


#ifdef _DEBUG
		std::vector<float> lpMean_h(this->layerData.lpMean.size());
		cudaMemcpy(&lpMean_h[0], thrust::raw_pointer_cast(&this->layerData.lpMean[0]), sizeof(float)*this->layerData.lpMean.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpVariance_h(this->layerData.lpVariance.size());
		cudaMemcpy(&lpVariance_h[0], thrust::raw_pointer_cast(&this->layerData.lpVariance[0]), sizeof(float)*this->layerData.lpVariance.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpDScale_h(this->lpDBias.size());
		cudaMemcpy(&lpDScale_h[0], thrust::raw_pointer_cast(&this->lpDScale[0]), sizeof(float)*lpDScale_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpDBias_h(this->lpDBias.size());
		cudaMemcpy(&lpDBias_h[0], thrust::raw_pointer_cast(&this->lpDBias[0]), sizeof(float)*lpDBias_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpScale_h(this->layerData.lpScale.size());
		cudaMemcpy(&lpScale_h[0], thrust::raw_pointer_cast(&this->layerData.lpScale[0]), sizeof(float)*lpScale_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpBias_h(this->layerData.lpBias.size());
		cudaMemcpy(&lpBias_h[0], thrust::raw_pointer_cast(&this->layerData.lpBias[0]), sizeof(float)*lpBias_h.size(), cudaMemcpyDeviceToHost);

#endif

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], this->m_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode BatchNormalization_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
