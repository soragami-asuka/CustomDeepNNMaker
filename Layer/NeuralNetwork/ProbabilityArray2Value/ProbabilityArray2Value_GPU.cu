//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"
#define _USE_MATH_DEFINES
#include<math.h>

#include"ProbabilityArray2Value_DATA.hpp"
#include"ProbabilityArray2Value_FUNC.hpp"
#include"ProbabilityArray2Value_Base.h"

#include"ProbabilityArray2Value_GPU.cuh"
#include"ProbabilityArray2Value_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)


	__global__ void device_Value2ProbabilityArray(
		U32 outputBatchBufferSize,
		F32 outputMinValue,
		F32 outputMaxValue,
		F32 variance,
		F32 lpDInputBuffer[],
		const F32 lpOutputBuffer[],
		const F32 lpDOutputBuffer[])
	{
		U32 batchNum  = blockIdx.x;
		U32 bufferPos = threadIdx.x;
		U32 inputCh = threadIdx.y;
		U32 inputChBufferSize = blockDim.x;
		U32 inputChSize = blockDim.y;

		U32 outputOffset = outputBatchBufferSize * batchNum + bufferPos;

		F32 trueValue = lpOutputBuffer[outputOffset] + lpDOutputBuffer[outputOffset];
		F32 value = (F32)inputCh / inputChSize
				  * (outputMaxValue - outputMinValue)
				  + outputMinValue;

		U32 inputOffset = (inputChBufferSize * inputChSize * batchNum) + (inputChBufferSize * inputCh) + bufferPos;

		lpDInputBuffer[inputOffset] = 1.0f / (2.0f * (F32)M_PI * variance) * expf(-(value - trueValue)*(value - trueValue) / (2.0f * variance * variance));
	}


	/** �R���X�g���N�^ */
	ProbabilityArray2Value_GPU::ProbabilityArray2Value_GPU(Gravisbell::GUID guid, ProbabilityArray2Value_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ProbabilityArray2Value_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	ProbabilityArray2Value_GPU::~ProbabilityArray2Value_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 ProbabilityArray2Value_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode ProbabilityArray2Value_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ProbabilityArray2Value_LayerData_Base& ProbabilityArray2Value_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ProbabilityArray2Value_LayerData_Base& ProbabilityArray2Value_GPU::GetLayerData()const
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
	ErrorCode ProbabilityArray2Value_GPU::PreProcessLearn()
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
	ErrorCode ProbabilityArray2Value_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ���͐M���̃`�����l�����Ƃ̃o�b�t�@�T�C�Y
		this->inputChannelSize = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		/**< ���͐M���̃o�b�`���Ƃ̃o�b�t�@�T�C�Y */
		this->inputBatchBufferSize = this->inputChannelSize * this->GetInputDataStruct().ch;

		// �ꎞ�o�̓o�b�t�@(�z�X�g������)
		this->lpTmpOutputBuffer_h.resize(this->outputBufferCount * this->GetBatchSize());
		this->lpTmpBatchOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->lpTmpBatchOutputBuffer_h[i] = &this->lpTmpOutputBuffer_h[this->outputBufferCount * i];

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode ProbabilityArray2Value_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode ProbabilityArray2Value_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// �o�̓o�b�t�@�̏�����
		cudaMemset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
		memset(&this->lpTmpOutputBuffer_h[0], 0, sizeof(F32)*this->lpTmpOutputBuffer_h.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 z=0; z<this->GetInputDataStruct().z; z++)
			{
				for(U32 y=0; y<this->GetInputDataStruct().y; y++)
				{
					for(U32 x=0; x<this->GetInputDataStruct().x; x++)
					{
						U32 offset = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, 0);

						// �ő�l�̔ԍ����擾
						S32 maxPos = -1;
						cublasIsamax_v2(
							this->cublasHandle,
							this->inputBatchBufferSize,
							&i_lppInputBuffer[this->inputBatchBufferSize*batchNum + offset],
							this->inputChannelSize,
							&maxPos);

						if(maxPos <= 0)
							continue;

						this->lpTmpBatchOutputBuffer_h[batchNum][offset]
							= (F32)(maxPos - 1) / this->GetInputDataStruct().ch
							* (this->layerData.layerStructure.outputMaxValue - this->layerData.layerStructure.outputMinValue)
							+ this->layerData.layerStructure.outputMinValue;
					}
				}
			}
		}

		// CPU > GPU
		cudaMemcpy(
			o_lppOutputBuffer,
			&this->lpTmpOutputBuffer_h[0],
			sizeof(F32) * this->lpTmpOutputBuffer_h.size(),
			cudaMemcpyHostToDevice);


#if _DEBUG
			std::vector<F32> lpInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpInputBuffer[0], i_lppInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode ProbabilityArray2Value_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// �o�̓o�b�t�@�̏�����
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

#if _DEBUG
			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], i_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			dim3 grid = this->GetBatchSize();
			dim3 block = dim3(this->inputChannelSize, this->GetInputDataStruct().ch);

			// ���K���z���v�Z
			device_Value2ProbabilityArray<<<grid, block>>>(
				this->outputBufferCount,
				this->layerData.layerStructure.outputMinValue,
				this->layerData.layerStructure.outputMaxValue,
				this->layerData.layerStructure.variance,
				o_lppDInputBuffer,
				i_lppOutputBuffer,
				i_lppDOutputBuffer);

#if _DEBUG
			std::vector<F32> lpTmpCalctBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTmpCalctBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// ���ω�
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufferPos=0; bufferPos<this->inputChannelSize; bufferPos++)
				{
					// ���v�����߂�
					F32 sumValue = 0.0f;
					cublasSasum_v2(
						this->cublasHandle,
						this->GetInputDataStruct().ch,
						&o_lppDInputBuffer[batchNum * this->inputBufferCount + bufferPos],
						this->inputChannelSize,
						&sumValue);

					// ���v�Ŋ���
					F32 alpha = 1.0f / sumValue;
					cublasSscal_v2(
						this->cublasHandle,
						this->GetInputDataStruct().ch,
						&alpha,
						&o_lppDInputBuffer[batchNum * this->inputBufferCount + bufferPos],
						this->inputChannelSize);
				}
			}


#if _DEBUG
			std::vector<F32> lpTeachBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTeachBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// �����Əo�͂Ō덷�����
			F32 alpha = -1;
			cublasSaxpy_v2(
				this->cublasHandle,
				this->inputBufferCount * this->GetBatchSize(),
				&alpha,
				i_lppInputBuffer,
				1,
				o_lppDInputBuffer,
				1);

#if _DEBUG
			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode ProbabilityArray2Value_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
