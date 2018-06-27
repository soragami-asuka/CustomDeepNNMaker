//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"MergeAdd_DATA.hpp"
#include"MergeAdd_FUNC.hpp"
#include"MergeAdd_Base.h"

#include"MergeAdd_GPU.cuh"
#include"MergeAdd_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define THREAD_PER_BLOCK	32

	/** ���͂𑫂����킹��.
		<outputChCount, batchSize> <32>
		@param	o_lpOutput			�o�̓o�b�t�@
		@param	i_outputChCount		�o�̓o�b�t�@��CH��
		@param	i_inputLyaerCount	���̓��C���[��
		@param	i_lppInput			���̓o�b�t�@
		@param	i_lpInputChCount	���̓o�b�t�@��CH��
		@param	i_bufferPerCh		�`�����l��������̃o�b�t�@��
		@param	i_loopCount			1�X���b�h������̎��s���[�v��
		*/
	__global__ void device_SumInput(F32* o_lpOutput, U32 i_outputChCount, U32 i_inputLayerCount, const F32*const* i_lppInput, const U32* i_lpInputChCount, U32 i_bufferPerCh, U32 i_loopCount, F32 i_scale)
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 outputOffset = (batchNum * i_outputChCount + chNum) * i_bufferPerCh + bufferPos;

			// �o�͏�����
			o_lpOutput[outputOffset] = 0.0f;
			for(U32 inputLayerNum=0; inputLayerNum<i_inputLayerCount; inputLayerNum++)
			{
				if(chNum >= i_lpInputChCount[inputLayerNum])
					continue;

				U32 inputOffset = (batchNum * i_lpInputChCount[inputLayerNum] + chNum) *i_bufferPerCh + bufferPos;

				o_lpOutput[outputOffset] += i_lppInput[inputLayerNum][inputOffset];
			}

			// �X�P�[�����|����
			o_lpOutput[outputOffset] *= i_scale;
		}
	}

	/** ���͌덷���v�Z����.
		<outputChCount, batchSize> <32>
		@param	o_lppDInput			���͌덷�o�b�t�@
		@param	i_lpInputChCount	���̓o�b�t�@��CH��
		@param	i_inputLyaerCount	���̓��C���[��
		@param	i_lpDOutput			�o�͌덷�o�b�t�@
		@param	i_bufferPerCh		�`�����l��������̃o�b�t�@��
		@param	i_loopCount			1�X���b�h������̎��s���[�v��
		*/
	__global__ void device_CalculateDInput(F32** o_lppDInput, const U32* i_lpInputChCount, U32 i_inputLayerCount, const F32* i_lpDOutput, U32 i_bufferPerCh, U32 i_loopCount, F32 i_scale)
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;
		U32 outputChCount = gridDim.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 outputOffset = (batchNum * outputChCount + chNum) * i_bufferPerCh + bufferPos;

			// ���͌덷�v�Z
			for(U32 inputLayerNum=0; inputLayerNum<i_inputLayerCount; inputLayerNum++)
			{
				if(chNum >= i_lpInputChCount[inputLayerNum])
					continue;

				U32 inputOffset = (batchNum * i_lpInputChCount[inputLayerNum] + chNum) *i_bufferPerCh + bufferPos;

				o_lppDInput[inputLayerNum][inputOffset] = i_lpDOutput[outputOffset] * i_scale;
			}
		}
	}


	/** �R���X�g���N�^ */
	MergeAdd_GPU::MergeAdd_GPU(Gravisbell::GUID guid, MergeAdd_LayerData_GPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeAdd_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	MergeAdd_GPU::~MergeAdd_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MergeAdd_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeAdd_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MergeAdd_LayerData_Base& MergeAdd_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeAdd_LayerData_Base& MergeAdd_GPU::GetLayerData()const
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
	ErrorCode MergeAdd_GPU::PreProcessLearn()
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
	ErrorCode MergeAdd_GPU::PreProcessCalculate()
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

		// CH������̃o�b�t�@��
		this->bufferCountPerCh = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		// �e���̓��C���[��CH��
		thrust::host_vector<U32> lpInputChCount(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			lpInputChCount[inputNum] = this->GetInputDataStruct(inputNum).ch;
		}
		this->lpInputChCount_d = lpInputChCount;

		// ���͐M���̐擪�A�h���X�̔z��
		// �o�b�t�@�̊m�ۂ̂�
		this->lppInputBuffer_d.resize(this->GetInputDataCount());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeAdd_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MergeAdd_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
#if 0
		// �o�̓o�b�t�@��������
		cudaMemset(&o_lppOutputBuffer[0], 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());

		F32 alpha = this->layerData.layerStructure.Scale;
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				cublasStatus_t err = cublasSaxpy_v2(
					this->cublasHandle,
					min(this->lpInputBufferCount[inputNum], outputBufferCount),
					&alpha,
					&i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]],
					1,
					&o_lppOutputBuffer[batchNum*this->outputBufferCount],
					1);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

			}
		}
		cudaThreadSynchronize();
#else
		// ���͐M���z���Device�ɃR�s�[
		cudaMemcpy(thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), i_lppInputBuffer, sizeof(F32*)*this->lppInputBuffer_d.size(), cudaMemcpyHostToDevice);

		// �v�Z
		dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_SumInput<<<grid,block>>>(
			o_lppOutputBuffer, this->GetOutputDataStruct().ch,
			this->GetInputDataCount(), thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), thrust::raw_pointer_cast(&this->lpInputChCount_d[0]),
			this->bufferCountPerCh,
			loopCount,
			this->layerData.layerStructure.Scale);
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
	ErrorCode MergeAdd_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
#if 0
			// ���͌덷�o�b�t�@�̏�����
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				cudaMemset(o_lppDInputBuffer[inputNum], 0, sizeof(F32)*this->lpInputBufferCount[inputNum]*this->GetBatchSize());
			}
			
			F32 alpha = this->layerData.layerStructure.Scale;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
				{
					cublasStatus_t err = cublasSaxpy_v2(
						this->cublasHandle,
						min(this->lpInputBufferCount[inputNum], outputBufferCount),
						&alpha,
						&i_lppDOutputBuffer[batchNum*this->outputBufferCount],
						1,
						&o_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
						1);

					if(err != 0)
						return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
				}
			}
			cudaThreadSynchronize();
#else
		// ���͌덷�M���z���Device�ɃR�s�[
		cudaMemcpy(thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), o_lppDInputBuffer, sizeof(F32*)*this->lppInputBuffer_d.size(), cudaMemcpyHostToDevice);

		// �v�Z
		dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_CalculateDInput<<<grid, block>>>(
			thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]),
			thrust::raw_pointer_cast(&this->lpInputChCount_d[0]), this->GetInputDataCount(),
			i_lppDOutputBuffer,
			this->bufferCountPerCh,
			loopCount,
			this->layerData.layerStructure.Scale);
#endif
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
	ErrorCode MergeAdd_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



} // Gravisbell;
} // Layer;
} // NeuralNetwork;
