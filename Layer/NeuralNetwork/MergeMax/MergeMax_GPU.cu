//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"MergeMax_DATA.hpp"
#include"MergeMax_FUNC.hpp"
#include"MergeMax_Base.h"

#include"MergeMax_GPU.cuh"
#include"MergeMax_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)

	__global__ void device_FillValue(U32 bufferCount, F32 lpOutputBuffer[], F32 value)
	{
		U32 batchNum = blockIdx.y;
		U32 bufNum   = blockIdx.x * blockDim.x + threadIdx.x;

		if(bufNum >= bufferCount)
			return;

		lpOutputBuffer[batchNum * bufferCount + bufNum] = value;
	}
	__global__ void device_CalculateMax(U32 maxBufferCount, U32 inputBufferCount, U32 outputBufferCount, const F32 lpInputBuffer[], F32 lpOutputBuffer[])
	{
		U32 batchNum = blockIdx.y;
		U32 bufNum   = blockIdx.x * blockDim.x + threadIdx.x;

		if(bufNum >= maxBufferCount)
			return;

		lpOutputBuffer[batchNum * outputBufferCount + bufNum] = max(lpOutputBuffer[batchNum * outputBufferCount + bufNum], lpInputBuffer[batchNum * inputBufferCount + bufNum]);
	}
	__global__ void device_CalculateDInput(U32 maxBufferCount, U32 inputBufferCount, U32 outputBufferCount, const F32 lpInputBuffer[], const F32 lpOutputBuffer[], F32 lpDInputBuffer[], const F32 lpDOutputBuffer[])
	{
		U32 batchNum = blockIdx.y;
		U32 bufNum   = blockIdx.x * blockDim.x + threadIdx.x;

		if(bufNum >= maxBufferCount)
			return;

		U32 inputPos  = batchNum * inputBufferCount  + bufNum;
		U32 outputPos = batchNum * outputBufferCount + bufNum;

		lpDInputBuffer[inputPos] = (lpOutputBuffer[outputPos] == lpInputBuffer[inputPos]) * lpDOutputBuffer[outputPos];
	}


	/** �R���X�g���N�^ */
	MergeMax_GPU::MergeMax_GPU(Gravisbell::GUID guid, MergeMax_LayerData_GPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeMax_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	MergeMax_GPU::~MergeMax_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MergeMax_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeMax_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MergeMax_LayerData_Base& MergeMax_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeMax_LayerData_Base& MergeMax_GPU::GetLayerData()const
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
	ErrorCode MergeMax_GPU::PreProcessLearn()
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
	ErrorCode MergeMax_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
			if(this->lpInputBufferCount[inputNum] == 0)
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;



		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeMax_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MergeMax_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// �o�̓o�b�t�@��������
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum+=CALC_BATCH_MAX)
		{
			dim3 grid(
				(this->outputBufferCount + (CALC_INPUT_MAX-1))/CALC_INPUT_MAX,
				min(this->GetBatchSize()-batchNum, CALC_BATCH_MAX));
			dim3 block(
				min(this->outputBufferCount, CALC_INPUT_MAX));

			device_FillValue<<<grid, block>>>(
				this->outputBufferCount,
				o_lppOutputBuffer,
				-FLT_MAX);
		}


#ifdef _DEBUG
		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], o_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif


		for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
		{
			U32 bufferSize = min(this->lpInputBufferCount[inputNum], this->outputBufferCount);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum+=CALC_BATCH_MAX)
			{
				dim3 grid(
					(bufferSize + (CALC_INPUT_MAX-1))/CALC_INPUT_MAX,
					min(this->GetBatchSize()-batchNum, CALC_BATCH_MAX));
				dim3 block(
					min(bufferSize, CALC_INPUT_MAX));

				device_CalculateMax<<<grid, block>>>(
					bufferSize,
					this->lpInputBufferCount[inputNum],
					this->outputBufferCount,
					i_lppInputBuffer[inputNum],
					o_lppOutputBuffer);
			}
			cudaThreadSynchronize();
		}


#ifdef _DEBUG
		std::vector<std::vector<float>> lpTmpInputBuffer(this->GetInputDataCount());
		for(int i=0; i<lpTmpInputBuffer.size(); i++)
		{
			lpTmpInputBuffer[i].resize(this->GetBatchSize() * this->lpInputBufferCount[i]);
			cudaMemcpy(&lpTmpInputBuffer[i][0], i_lppInputBuffer[i], sizeof(float)*lpTmpInputBuffer[i].size(), cudaMemcpyDeviceToHost);
		}

		cudaMemcpy(&lpTmpOutputBuffer[0], o_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif


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
	ErrorCode MergeMax_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̏�����
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				cudaMemset(o_lppDInputBuffer[inputNum], 0, sizeof(F32)*this->lpInputBufferCount[inputNum]*this->GetBatchSize());
			}


			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				U32 bufferSize = min(this->lpInputBufferCount[inputNum], this->outputBufferCount);

				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum+=CALC_BATCH_MAX)
				{
					dim3 grid(
						(bufferSize + (CALC_INPUT_MAX-1))/CALC_INPUT_MAX,
						min(this->GetBatchSize()-batchNum, CALC_BATCH_MAX));
					dim3 block(
						min(bufferSize, CALC_INPUT_MAX));

					device_CalculateDInput<<<grid, block>>>(
						bufferSize,
						this->lpInputBufferCount[inputNum],
						this->outputBufferCount,
						i_lppInputBuffer[inputNum],
						i_lppOutputBuffer,
						o_lppDInputBuffer[inputNum],
						i_lppDOutputBuffer);
				}

				cudaThreadSynchronize();
			}
		}


#ifdef _DEBUG
		std::vector<std::vector<float>> lpTmpInputBuffer(this->GetInputDataCount());
		for(int i=0; i<lpTmpInputBuffer.size(); i++)
		{
			lpTmpInputBuffer[i].resize(this->GetBatchSize() * this->lpInputBufferCount[i]);
			cudaMemcpy(&lpTmpInputBuffer[i][0], i_lppInputBuffer[i], sizeof(float)*lpTmpInputBuffer[i].size(), cudaMemcpyDeviceToHost);
		}

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<std::vector<float>> lpTmpDInputBuffer(this->GetInputDataCount());
		for(int i=0; i<lpTmpInputBuffer.size(); i++)
		{
			lpTmpDInputBuffer[i].resize(this->GetBatchSize() * this->lpInputBufferCount[i]);
			cudaMemcpy(&lpTmpDInputBuffer[i][0], o_lppDInputBuffer[i], sizeof(float)*lpTmpDInputBuffer[i].size(), cudaMemcpyDeviceToHost);
		}
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode MergeMax_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



} // Gravisbell;
} // Layer;
} // NeuralNetwork;
