//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

#include"Convolution_GPU.cuh"
#include"Convolution_LayerData_GPU.cuh"

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define TEMPORARY_MEMORY_MAX	(100 * 1024 * 1024)

#define WORKSPACE_CODE			L"WorkSpace"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Convolution_GPU::Convolution_GPU(Gravisbell::GUID guid, Convolution_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Convolution_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData			(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount	(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount			(0)		/**< �j���[������ */
		,	outputBufferCount	(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
		,	biasTensorDesc		(NULL)
		,	filterDesc			(NULL)
		,	convDesc			(NULL)
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateTensorDescriptor(&biasTensorDesc);
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
	}
	/** �f�X�g���N�^ */
	Convolution_GPU::~Convolution_GPU()
	{
		if(convDesc)			cudnnDestroyConvolutionDescriptor(convDesc);
		if(filterDesc)			cudnnDestroyFilterDescriptor(filterDesc);
		if(biasTensorDesc)		cudnnDestroyTensorDescriptor(biasTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Convolution_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Convolution_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Convolution_LayerData_Base& Convolution_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Convolution_LayerData_Base& Convolution_GPU::GetLayerData()const
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
	ErrorCode Convolution_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;


		// �p�����[�^�ω��ʂ̃o�b�t�@���m��
		this->lpDBias.resize(this->layerData.pWeightData->GetBiasSize());
		this->lpDNeuron.resize(this->layerData.pWeightData->GetWeigthSize());


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->layerData.layerStructure.Output_Channel;
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		cudnnStatus_t err_cudnn;

		// �������𒲂ׂ�
		S32 dataDim = 1 + 1 + 0;	// �o�b�` + �`�����l�� + ����0
		std::vector<S32> dimInput;			// ���̓f�[�^�\��
		std::vector<S32> dimInputStride;	// ���̓f�[�^�̊e�������Ƃ̃f�[�^��
		std::vector<S32> dimBias;
		std::vector<S32> dimBiasStride;
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// �t�B���^������	���̓`�����l�� + �o�̓`�����l�� + ����
		std::vector<S32> dimFilter;
		S32 convDim = 0;			// ��ݍ��ݎ�����	����
		std::vector<S32> dimStride;
		std::vector<S32> dimUpscale;
		std::vector<S32> dimPadding;
		if(this->GetInputDataStruct().z > 1)
		{
			dataDim = 1 + 1 + 3;

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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;
			dimBias[4] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[1] = dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[2] = dimBias[3] * dimBias[4];
			dimBiasStride[3] = dimBias[4];
			dimBiasStride[4] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 3;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.z;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[4] = this->layerData.layerStructure.FilterSize.x;

			convDim = 3;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.z;
			dimPadding[1] = this->layerData.layerStructure.Padding.y;
			dimPadding[2] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.z;
			dimUpscale[1] = this->layerData.layerStructure.Dilation.y;
			dimUpscale[2] = this->layerData.layerStructure.Dilation.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.z;
			dimStride[1] = this->layerData.layerStructure.Stride.y;
			dimStride[2] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->GetInputDataStruct().y > 1 || this->GetInputDataStruct().x >= 1)
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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3];
			dimBiasStride[1] = dimBias[2] * dimBias[3];
			dimBiasStride[2] = dimBias[3];
			dimBiasStride[3] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 2;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.x;

			convDim = 2;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.y;
			dimPadding[1] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.y;
			dimUpscale[1] = this->layerData.layerStructure.Dilation.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.y;
			dimStride[1] = this->layerData.layerStructure.Stride.x;
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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
			dimBias[2] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2];
			dimBiasStride[1] = dimBias[2];
			dimBiasStride[2] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 1;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.x;

			convDim = 1;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.x;
		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// CUDNN�̓��̓f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// �t�B���^�T�C�Y��ݒ�
		err_cudnn = cudnnSetFilterNdDescriptor(
			this->filterDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			filterDim,
			&dimFilter[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// ��ݍ��ݏ����ݒ�
		err_cudnn = cudnnSetConvolutionNdDescriptor(
			this->convDesc,
			convDim,
			&dimPadding[0],
			&dimStride[0],
			&dimUpscale[0],
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �o�̓f�[�^�\�����擾
        err_cudnn = cudnnGetConvolutionNdForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			dataDim,
			&dimOutput[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̏o�̓f�[�^�\����Gravisbell�̏o�̓f�[�^�\������v���邱�Ƃ��m�F
		Gravisbell::Vector3D<S32> outputVector;
		S32 outputBatchSize = dimOutput[0];
		S32 outputCh = dimOutput[1];
		if(dataDim == 5)
		{
			outputVector.z = dimOutput[2];
			outputVector.y = dimOutput[3];
			outputVector.x = dimOutput[4];
		}
		else if(dataDim == 4)
		{
			outputVector.z = 1;
			outputVector.y = dimOutput[2];
			outputVector.x = dimOutput[3];
		}
		else if(dataDim == 3)
		{
			outputVector.z = 1;
			outputVector.y = 1;
			outputVector.x = dimOutput[2];
		}
		if(outputBatchSize != this->GetBatchSize())
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputCh != this->GetOutputDataStruct().ch)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.z != this->GetOutputDataStruct().z)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.y != this->GetOutputDataStruct().y)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.x != this->GetOutputDataStruct().x)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̏o�̓f�[�^�\����ݒ�
		dimOutputStride.resize(dataDim);
		for(S32 i=0; i<dataDim; i++)
		{
			dimOutputStride[i] = 1;
			for(S32 j=i+1; j<dataDim; j++)
				dimOutputStride[i] *= dimOutput[j];
		}
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// �ő��̃A���S���Y������������(�O���`�d)
		err_cudnn = cudnnGetConvolutionForwardAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			cudnnConvolutionFwdPreference_t::CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			TEMPORARY_MEMORY_MAX,										// �g�p�\�ȃ������̏��
			&this->useForwardAlgorithm
			);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(�O���`�d)
		size_t workSpaceSizeByte_forward;
		err_cudnn = cudnnGetConvolutionForwardWorkspaceSize(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->useForwardAlgorithm,
			&workSpaceSizeByte_forward);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �ő��̃A���S���Y������������(����`�d-�f�[�^)
		err_cudnn = cudnnGetConvolutionBackwardDataAlgorithm(
			this->cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			cudnnConvolutionBwdDataPreference_t::CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			TEMPORARY_MEMORY_MAX,																				// �g�p�\�ȃ������̏��
			&this->useBackwardDataAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(����`�d-�f�[�^)
		size_t workSpaceSizeByte_backwardData;
		err_cudnn = cudnnGetConvolutionBackwardDataWorkspaceSize(
			this->cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->useBackwardDataAlgorithm,
			&workSpaceSizeByte_backwardData);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �ő��̃A���S���Y������������(����`�d-�f�[�^)
		err_cudnn = cudnnGetConvolutionBackwardFilterAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			cudnnConvolutionBwdFilterPreference_t::CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			TEMPORARY_MEMORY_MAX,																					// �g�p�\�ȃ������̏��
			&this->useBackwardFilterAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(����`�d-�f�[�^)
		size_t workSpaceSizeByte_backwardFilter;
		err_cudnn = cudnnGetConvolutionBackwardFilterWorkspaceSize(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->useBackwardFilterAlgorithm,
			&workSpaceSizeByte_backwardFilter);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �����p�o�b�t�@�̊m��
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), WORKSPACE_CODE, (U32)max(workSpaceSizeByte_forward, max(workSpaceSizeByte_backwardData, workSpaceSizeByte_backwardFilter)));

		// �o�C�A�X�̃f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->biasTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimBias[0],
			&dimBiasStride[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Convolution_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			// ���Ƃ肠����CPU���ŏ���.
			// ��{�I��1�񂵂��ʂ�Ȃ����珈�����ׂɉe���͗^���Ȃ��E�E�E�͂�
			// ���蔲��


			U32 PROCTIME_MAX = 5;			// ���s�ő�l
			F32	VARIANCE_TOLERANCE = 0.1f;	// ���U����(���e�͈�)

			std::vector<F32> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);

			// �o�b�t�@���m��
			thrust::device_vector<F32> lpTmpWeight_d(this->layerData.pWeightData->GetWeigthSize());
			thrust::device_vector<F32> lpTmpBias_d(this->layerData.pWeightData->GetBiasSize());

			// �o�b�t�@���R�s�[
			cudaMemcpy(thrust::raw_pointer_cast(&lpTmpWeight_d[0]), this->layerData.pWeightData->GetWeight(), sizeof(F32)*lpTmpWeight_d.size(), cudaMemcpyDeviceToDevice);
			cudaMemcpy(thrust::raw_pointer_cast(&lpTmpBias_d[0]),   this->layerData.pWeightData->GetBias(),   sizeof(F32)*lpTmpBias_d.size(), cudaMemcpyDeviceToDevice);


			U32 procTime = 0;
			do
			{
				// ���Z�����s
				ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer, thrust::raw_pointer_cast(&lpTmpWeight_d[0]), thrust::raw_pointer_cast(&lpTmpBias_d[0]));
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// �o�b�t�@���R�s�[
				cudaMemcpy(&lpTmpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

				// �o�͂̕��U�����߂�
				F32 variance = 0.0f;
				F32 average  = 0.0f;
				{
					// ���ς����߂�
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						average += lpTmpOutputBuffer[outputNum];
					}
					average /= lpTmpOutputBuffer.size();

					// ���U�����߂�
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						variance += (lpTmpOutputBuffer[outputNum] - average) * (lpTmpOutputBuffer[outputNum] - average);
					}
					variance /= lpTmpOutputBuffer.size();
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// �W���΍��ŏd�݂������čX�V����
				F32 deviation = sqrtf(variance);
				{
					thrust::host_vector<F32> lpTmpNeuron = lpTmpWeight_d;
					thrust::host_vector<F32> lpTmpBias   = lpTmpBias_d;

					for(U32 neuronNum=0; neuronNum<lpTmpNeuron.size(); neuronNum++)
					{
						lpTmpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<lpTmpBias.size(); neuronNum++)
					{
						lpTmpBias[neuronNum] /= deviation;
					}

					lpTmpWeight_d = lpTmpNeuron;
					lpTmpBias_d    = lpTmpBias;
				}

				procTime++;
			}while(procTime < 5);
			
			// �d�݂��X�V
			this->layerData.pWeightData->SetData(thrust::raw_pointer_cast(&lpTmpWeight_d[0]), thrust::raw_pointer_cast(&lpTmpBias_d[0]));
		}
		else
		{
			ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer, this->layerData.pWeightData->GetWeight(), this->layerData.pWeightData->GetBias());
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode Convolution_GPU::Calculate_base(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer, const F32* lpWeight, const F32* lpBias)
	{
		cudnnStatus_t err_cudnn;

		// ��ݍ��ݏ���
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;
			err_cudnn = cudnnConvolutionForward(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->filterDesc,
				lpWeight,
				this->convDesc,
				this->useForwardAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �o�C�A�X��ǉ�
		{
			F32 alpha = 1.0f;
			F32 beta  = 1.0f;

			err_cudnn = cudnnAddTensor(
				this->cudnnHandle,
				&alpha,
				this->biasTensorDesc,
				lpBias,
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �ꎞ�o�b�t�@���
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

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
	ErrorCode Convolution_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���͌덷���v�Z
		if(o_lppDInputBuffer)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				this->layerData.pWeightData->GetWeight(),
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->inputTensorDesc,
				o_lppDInputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �ꎞ�o�b�t�@���
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Convolution_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// �t�B���^�[�ω��ʂ��v�Z
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardFilter(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				this->convDesc,
				this->useBackwardFilterAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->lpDNeuron[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �o�C�A�X�ω��ʂ��v�Z
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardBias(
				this->cudnnHandle,
				&alpha,
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				&beta,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->lpDBias[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �ω��ʂ𔽉f
		this->layerData.pWeightData->UpdateData(thrust::raw_pointer_cast(&this->lpDNeuron[0]), thrust::raw_pointer_cast(&this->lpDBias[0]));

		// �ꎞ�o�b�t�@���
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);


		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
