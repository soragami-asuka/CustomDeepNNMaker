//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"LimitBackPropagationRange_DATA.hpp"
#include"LimitBackPropagationRange_FUNC.hpp"
#include"LimitBackPropagationRange_Base.h"

#include"LimitBackPropagationRange_GPU.cuh"
#include"LimitBackPropagationRange_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
#define THREAD_PER_BLOCK	32
#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)
	
	__global__ void device_ReLimitBackPropagationRange(
		U32 chCount,
		U32 startX, U32 startY, U32 startZ,
		U32 inputXCount,  U32 inputYCount,  U32 inputZCount,
		U32 outputXCount, U32 outputYCount, U32 outputZCount,
		U32 i_bufferPerCh,
		U32 i_loopCount,
		const F32 lpDOutputBuffer[],
		F32 lpDInputBuffer[])
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 x = bufferPos % inputXCount;
			U32 y = bufferPos / inputXCount % inputYCount;
			U32 z = bufferPos / inputXCount / inputYCount % inputZCount;

			if(x < startX || x >= startX + outputXCount)
				continue;
			if(y < startY || y >= startY + outputYCount)
				continue;
			if(z < startZ || z >= startZ + outputZCount)
				continue;

			U32 offset = batchNum * (chCount * i_bufferPerCh) + chNum * i_bufferPerCh + bufferPos;

			lpDInputBuffer[offset] = lpDOutputBuffer[offset];
		}
	}

	/** �R���X�g���N�^ */
	LimitBackPropagationRange_GPU::LimitBackPropagationRange_GPU(Gravisbell::GUID guid, LimitBackPropagationRange_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	LimitBackPropagationRange_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	LimitBackPropagationRange_GPU::~LimitBackPropagationRange_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 LimitBackPropagationRange_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode LimitBackPropagationRange_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	LimitBackPropagationRange_LayerData_Base& LimitBackPropagationRange_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const LimitBackPropagationRange_LayerData_Base& LimitBackPropagationRange_GPU::GetLayerData()const
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
	ErrorCode LimitBackPropagationRange_GPU::PreProcessLearn()
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
	ErrorCode LimitBackPropagationRange_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// CH������̃o�b�t�@��
		this->bufferCountPerCh = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LimitBackPropagationRange_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode LimitBackPropagationRange_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ���̓o�b�t�@���o�̓o�b�t�@�ɃR�s�[
		cudaMemcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetBatchSize()*this->outputBufferCount, cudaMemcpyDeviceToDevice);

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
	ErrorCode LimitBackPropagationRange_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// �o�̓o�b�t�@�̏�����
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
			
			// �v�Z
			dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
			dim3 block(THREAD_PER_BLOCK);
			U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

			device_ReLimitBackPropagationRange<<<grid, block>>>(
				this->GetOutputDataStruct().ch,
				this->layerData.layerStructure.startPosition.x, this->layerData.layerStructure.startPosition.y, this->layerData.layerStructure.startPosition.z,
				this->GetInputDataStruct().x, this->GetInputDataStruct().y, this->GetInputDataStruct().z,
				this->layerData.layerStructure.boxSize.x, this->layerData.layerStructure.boxSize.y, this->layerData.layerStructure.boxSize.z,
				this->bufferCountPerCh, loopCount,
				i_lppDOutputBuffer,
				o_lppDInputBuffer);
			
#if _DEBUG
			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode LimitBackPropagationRange_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
