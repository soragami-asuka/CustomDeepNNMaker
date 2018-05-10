//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"Value2SignalArray_DATA.hpp"
#include"Value2SignalArray_FUNC.hpp"
#include"Value2SignalArray_Base.h"

#include"Value2SignalArray_GPU.cuh"
#include"Value2SignalArray_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define WORKSPACE_CODE			L"WorkSpace"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)

	__global__ void device_Value2SignalArray(
		U32 resolution,
		F32 inputMinValue,
		F32 inputMaxValue,
		const F32 lpInputBuffer[],
		F32 lpOutputBuffer[])
	{
		U32 batchNum     = blockIdx.x;
		U32 inputCh      = blockIdx.y;
		U32 inputChCount = gridDim.y;
		U32 bufferPos    = threadIdx.x;
		U32 inputChBufferSize = blockDim.x;

		U32 inputOffset = batchNum * inputChCount * inputChBufferSize + inputCh * inputChBufferSize + bufferPos;
		F32 inputValue  = lpInputBuffer[inputOffset];

		// �o�̓`�����l���ԍ���float�Ōv�Z
		F32 fOutputCh = max(0.0f, min((F32)resolution-1 - 1e-8, (inputValue - inputMinValue) / (inputValue - inputMinValue) * resolution));

		// �����l�ɕϊ�
		U32 iOutputCh = (U32)fOutputCh;
		F32 t = fOutputCh - iOutputCh;

		U32 outputOffset0 = batchNum * (inputChCount*resolution) * inputChBufferSize + (inputCh*resolution + iOutputCh + 0) * inputChBufferSize + bufferPos;
		U32 outputOffset1 = batchNum * (inputChCount*resolution) * inputChBufferSize + (inputCh*resolution + iOutputCh + 1) * inputChBufferSize + bufferPos;

		lpOutputBuffer[outputOffset0] = (1.0f - t);
		lpOutputBuffer[outputOffset0] = t;
	}


	/** �R���X�g���N�^ */
	Value2SignalArray_GPU::Value2SignalArray_GPU(Gravisbell::GUID guid, Value2SignalArray_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Value2SignalArray_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	Value2SignalArray_GPU::~Value2SignalArray_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Value2SignalArray_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Value2SignalArray_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Value2SignalArray_LayerData_Base& Value2SignalArray_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Value2SignalArray_LayerData_Base& Value2SignalArray_GPU::GetLayerData()const
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
	ErrorCode Value2SignalArray_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// Signal -> value�ϊ����s�����߂̏d�ݔz��̍쐬
		std::vector<F32> lpSignal2ValueWeight_h(this->layerData.layerStructure.resolution);
		for(U32 i=0; i<this->layerData.layerStructure.resolution; i++)
		{
			lpSignal2ValueWeight_h[i] = (this->layerData.layerStructure.inputMaxValue - this->layerData.layerStructure.inputMinValue) * i / this->layerData.layerStructure.resolution - this->layerData.layerStructure.inputMinValue;
		}

		lpSignal2ValueWeight_d.resize(this->layerData.layerStructure.resolution);
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpSignal2ValueWeight_d[0]), &lpSignal2ValueWeight_h[0], sizeof(F32)*lpSignal2ValueWeight_h.size(), cudaMemcpyHostToDevice);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Value2SignalArray_GPU::PreProcessCalculate()
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

		// �����p�o�b�t�@�̊m��
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), WORKSPACE_CODE, sizeof(F32)*this->GetBatchSize()*this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Value2SignalArray_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Value2SignalArray_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// �o�̓o�b�t�@�̏�����
		cudaMemset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
		memset(&this->lpTmpOutputBuffer_h[0], 0, sizeof(F32)*this->lpTmpOutputBuffer_h.size());

		dim3 grid(this->GetBatchSize(), this->GetInputDataStruct().ch);
		dim3 block(this->inputChannelSize);

		device_Value2SignalArray<<<grid, block>>>(
			this->layerData.layerStructure.resolution,
			this->layerData.layerStructure.inputMinValue,
			this->layerData.layerStructure.inputMaxValue,
			i_lppInputBuffer,
			o_lppOutputBuffer);

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
	ErrorCode Value2SignalArray_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// ���̓o�b�t�@����͌덷�o�b�t�@�ɃR�s�[
			cudaMemcpy(o_lppDInputBuffer, i_lppInputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);

			// ���t�f�[�^�쐬�p�̈ꎞ�o�b�t�@���擾
			F32* lpSignalTeachBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE);

			// signal�̋��t�f�[�^���쐬
			cudaMemcpy(lpSignalTeachBuffer, i_lppOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			F32 alpha = 1;
			cublasSaxpy_v2(
				this->cublasHandle,
				this->outputBufferCount * this->GetBatchSize(),
				&alpha,
				i_lppDOutputBuffer,
				1,
				lpSignalTeachBuffer,
				1);

			// signal -> value�ϊ����ďo�͌덷�o�b�t�@�Ɋi�[
			alpha =  1.0f;
			F32 beta  = -1.0f;	// y�ɂ͓��̓o�b�t�@�������Ă��邽�߁A���t�M��-���͐M���ɂ���
			cublasSgemv_v2(
				this->cublasHandle,
				CUBLAS_OP_N,
				this->GetOutputDataStruct().ch,
				this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z,
				&alpha,
				i_lppOutputBuffer,
				this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z,
				thrust::raw_pointer_cast(&this->lpSignal2ValueWeight_d[0]),
				1,
				&beta,
				o_lppDInputBuffer,
				1);

#if _DEBUG
			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], i_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpTeachBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTeachBuffer[0], lpSignalTeachBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// ���t�f�[�^�o�b�t�@�̊J��
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Value2SignalArray_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
