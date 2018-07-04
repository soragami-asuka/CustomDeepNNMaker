//======================================
// �o�b�`���K�����C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"ExponentialNormalization_DATA.hpp"
#include"ExponentialNormalization_FUNC.hpp"
#include"ExponentialNormalization_Base.h"

#include"ExponentialNormalization_GPU.cuh"
#include"ExponentialNormalization_LayerData_GPU.cuh"

#define WORKSPACE_CODE			L"WorkSpace"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
#define CALCULATE_DSCALE_BLOCK_SIZE	32

	/** CH���Ƃ̕��ς����߂�.
		dim = <ch,1,1>
		block = <32,1,1>
	*/
	__global__ void device_UpdateChAverage(F32* o_lpAverage, const F32* i_lpInputValue, U32 i_inputCountPerChannel, U32 i_batchSize, U32 i_loopCount, F32 i_alpha)
	{
		__shared__ F32 lpTmpSumValue[CALCULATE_DSCALE_BLOCK_SIZE];

		U32 chNum = blockIdx.x;
		U32 chCount = gridDim.x;
		U32 tid = threadIdx.x;
		U32 inputCount = chCount * i_inputCountPerChannel;

		// DWeight��Vector�̏�Z���v�Z
		lpTmpSumValue[tid] = 0.0f;
		for(U32 batchNum=0; batchNum<i_batchSize; batchNum++)
		{
			for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
			{
				U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
				if(inputNum >= i_inputCountPerChannel)
					continue;

				U32 offset = batchNum * inputCount + chNum * i_inputCountPerChannel + inputNum;

				lpTmpSumValue[tid] += i_lpInputValue[offset];
			}
		}
		__syncthreads();

		// ���v
		lpTmpSumValue[tid] += lpTmpSumValue[tid + 16];
		__syncthreads();
		lpTmpSumValue[tid] += lpTmpSumValue[tid + 8];
		__syncthreads();
		lpTmpSumValue[tid] += lpTmpSumValue[tid + 4];
		__syncthreads();
		lpTmpSumValue[tid] += lpTmpSumValue[tid + 2];
		__syncthreads();
		lpTmpSumValue[tid] += lpTmpSumValue[tid + 1];
		__syncthreads();

		if(tid == 0)
		{
			F32 average = lpTmpSumValue[tid] / (i_inputCountPerChannel * i_batchSize);

			o_lpAverage[chNum] = i_alpha * average + (1.0f - i_alpha) * o_lpAverage[chNum];
		}
	}

	/** �R���X�g���N�^ */
	ExponentialNormalization_GPU::ExponentialNormalization_GPU(Gravisbell::GUID guid, ExponentialNormalization_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ExponentialNormalization_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount		(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount		(0)				/**< �o�̓o�b�t�@�� */
		,	channeclBufferCount		(0)				/**< 1�`�����l��������̃o�b�t�@�� */
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	ExponentialNormalization_GPU::~ExponentialNormalization_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 ExponentialNormalization_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode ExponentialNormalization_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ExponentialNormalization_LayerData_Base& ExponentialNormalization_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ExponentialNormalization_LayerData_Base& ExponentialNormalization_GPU::GetLayerData()const
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
	ErrorCode ExponentialNormalization_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �ꎞ�o�b�t�@�̃T�C�Y�����߂�
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), WORKSPACE_CODE, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode ExponentialNormalization_GPU::PreProcessCalculate()
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
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode ExponentialNormalization_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode ExponentialNormalization_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
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
	ErrorCode ExponentialNormalization_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���͌덷�o�b�t�@�̃A�h���X���i�[
		if(o_lppDInputBuffer == NULL)
		{
			// ���͌덷�o�b�t�@�����݂��Ȃ��ꍇ�w�K���ł��Ȃ����߁A��փo�b�t�@���m��
			o_lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE);
		}



#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		// �ꎞ�o�b�t�@���J��
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode ExponentialNormalization_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���͌덷�o�b�t�@�̃A�h���X���i�[
		if(o_lppDInputBuffer == NULL)
		{
			// ���͌덷�o�b�t�@�����݂��Ȃ��ꍇ�w�K���ł��Ȃ����߁A��փo�b�t�@���m��
			o_lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE);
		}


#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		// �ꎞ�o�b�t�@���J��
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
