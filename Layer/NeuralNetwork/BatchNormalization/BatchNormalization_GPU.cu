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
	BatchNormalization_GPU::BatchNormalization_GPU(Gravisbell::GUID guid, BatchNormalization_LayerData_GPU& i_layerData)
		:	BatchNormalization_Base	(guid)
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
	ErrorCode BatchNormalization_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�p�̕ϐ����쐬
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->layerData.inputDataStruct.ch, 0.0f);
		this->lpTmpVariance.resize(this->layerData.inputDataStruct.ch, 0.0f);

		// ���͌덷�p�o�b�t�@���쐬
		this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		cudnnStatus_t err_cudnn;

		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �`�����l�����Ƃ̃o�b�t�@�����m�F
		this->channeclBufferCount = this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);


		// �������𒲂ׂ�
		S32 dataDim = 1 + 1 + 0;	// �o�b�` + �`�����l�� + ����0
		std::vector<S32> dimInput;			// ���̓f�[�^�\��
		std::vector<S32> dimInputStride;	// ���̓f�[�^�̊e�������Ƃ̃f�[�^��
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		std::vector<S32> dimParam;
		std::vector<S32> dimParamStride;
		if(this->layerData.inputDataStruct.z > 1)
		{
			dataDim = 1 + 1 + 3;	// �o�b�` + �`�����l�� + ����3

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.z;
			dimInput[3] = this->layerData.inputDataStruct.y;
			dimInput[4] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.GetOutputDataStruct().ch;
			dimOutput[2] = this->layerData.GetOutputDataStruct().z;
			dimOutput[3] = this->layerData.GetOutputDataStruct().y;
			dimOutput[4] = this->layerData.GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[2] = dimOutput[3] * dimOutput[4];
			dimOutputStride[3] = dimOutput[4];
			dimOutputStride[4] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->layerData.inputDataStruct.ch;
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
		else if(this->layerData.inputDataStruct.y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.y;
			dimInput[3] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.GetOutputDataStruct().ch;
			dimOutput[2] = this->layerData.GetOutputDataStruct().y;
			dimOutput[3] = this->layerData.GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->layerData.inputDataStruct.ch;
			dimParam[2] = 1;
			dimParam[3] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2] * dimParam[3];
			dimParamStride[1] = dimParam[2] * dimParam[3];
			dimParamStride[2] = dimParam[3];
			dimParamStride[3] = 1;
		}
		else if(this->layerData.inputDataStruct.x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.GetOutputDataStruct().ch;
			dimOutput[2] = this->layerData.GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[3];
			dimOutputStride[2] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->layerData.inputDataStruct.ch;
			dimParam[2] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[2] * dimParam[3];
			dimParamStride[1] = dimParam[3];
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


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		// �w�K�ݒ��ۑ�
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		this->pLearnData->WriteToStruct((BYTE*)&learnData);


		// �w�K�񐔂�������
		this->learnCount = 0;

		// ���Z�p�̕���.���U��������
		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			this->layerData.lpMean[ch] = 0.0f;
			this->layerData.lpVariance[ch] = 0.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_GPU::PreProcessCalculateLoop()
	{
		// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

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

			std::vector<F32> lpVarianceLast(this->layerData.inputDataStruct.ch);
			for(U32 i=0; i<lpVarianceLast.size(); i++)
				lpVarianceLast[i] = this->layerData.lpVariance[i];

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
				thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
				thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
				max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
				thrust::raw_pointer_cast(&this->lpTmpMean[0]),
				thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

			std::vector<F32> lpVarianceNext(this->layerData.inputDataStruct.ch);
			for(U32 i=0; i<lpVarianceNext.size(); i++)
				lpVarianceNext[i] = this->layerData.lpVariance[i];

			std::vector<F32> lpVarianceTmp(this->layerData.inputDataStruct.ch);
			for(U32 i=0; i<lpVarianceTmp.size(); i++)
				lpVarianceTmp[i] = this->lpTmpVariance[i];

			// �w�K�����̎��s�񐔂��J�E���g�A�b�v
			this->learnCount++;
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
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode BatchNormalization_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		cudnnStatus_t err_cudnn;

		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBufferPrev = i_lpDOutputBufferPrev;

		F32 alphaData = 1.0f;
		F32 betaData  = 0.0f;

		F32 alphaParam = this->learnData.LearnCoeff;
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
			thrust::raw_pointer_cast(&this->lpDInputBuffer[0]),
			this->paramTensorDesc,
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
			max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
			thrust::raw_pointer_cast(&this->lpTmpMean[0]),
			thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_GPU::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
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
