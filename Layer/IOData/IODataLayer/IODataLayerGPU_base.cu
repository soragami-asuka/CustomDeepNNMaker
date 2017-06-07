//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
//===================================


#include "stdafx.h"
#include "IODataLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>


#define BLOCK_SIZE	(16)

using namespace Gravisbell;

namespace
{
	/** �x�N�g���̗v�f���m�̊|���Z. */
	__global__ void cuda_func_calculateError(const F32* i_lpOutputBuffer, const F32* i_lpTeachBuffer, F32* o_lpErrorMax, F32* o_lpErrorAve, F32* o_lpErrorAve2, F32* o_lpErrorCrossEntropy, U32 i_bachNum, U32 i_bufferSize)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;

		const U32 bufferPos = i_bachNum * i_bufferSize + inputNum;

		F32 teach = i_lpTeachBuffer[bufferPos];
		F32 output = i_lpOutputBuffer[bufferPos];

		F32 error = (teach - output);
		F32 error_abs = abs(error);

		F32 crossEntropy = -(F32)(
			      teach  * log(max(0.0001,  output)) +
				 (1 - teach) * log(max(0.0001,1-output))
				 );

		// �덷��ۑ�
		o_lpErrorMax[inputNum]  = max(o_lpErrorMax[inputNum], error_abs);
		o_lpErrorAve[inputNum]  += error_abs;
		o_lpErrorAve2[inputNum] += error_abs * error_abs;
		o_lpErrorCrossEntropy[inputNum] += crossEntropy;
	}
}



namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** �R���X�g���N�^ */
	IODataLayerGPU_base::IODataLayerGPU_base(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
		:	guid				(guid)
		,	ioDataStruct		(ioDataStruct)
		,	lpBatchDataNoList	(NULL)
		,	calcErrorCount		(0)
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	IODataLayerGPU_base::~IODataLayerGPU_base()
	{
		cublasDestroy(cublasHandle);
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode IODataLayerGPU_base::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//==============================
	// ���C���[���ʌn
	//==============================
	/** ���C���[��ʂ̎擾 */
	U32 IODataLayerGPU_base::GetLayerKind()const
	{
		return ELayerKind::LAYER_KIND_GPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
	}

	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID IODataLayerGPU_base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID IODataLayerGPU_base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		Gravisbell::Layer::IOData::GetLayerCode(layerCode);

		return layerCode;
	}
	
	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* IODataLayerGPU_base::GetLayerStructure()const
	{
		return NULL;
	}

	//==============================
	// �f�[�^�Ǘ��n
	//==============================
	/** �f�[�^�̍\�������擾���� */
	IODataStruct IODataLayerGPU_base::GetDataStruct()const
	{
		return this->ioDataStruct;
	}

	/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
		@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����F32�^�z��̗v�f��. */
	U32 IODataLayerGPU_base::GetBufferCount()const
	{
		return this->ioDataStruct.GetDataCount();
	}


	//==============================
	// ���C���[���ʌn
	//==============================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessLearn(U32 batchSize)
	{
		// �ʏ�̉��Z�p�̏��������s
		ErrorCode err = PreProcessCalculate(batchSize);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// �덷�����f�[�^�z��̏�����
		this->lpDInputBuffer.resize(batchSize * this->GetBufferCount());

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessCalculate(U32 batchSize)
	{
		// �o�b�`�T�C�Y�̕ۑ�
		this->batchSize = batchSize;

		// �o�b�t�@�̊m�ۂƃo�b�`�����f�[�^�z��̏�����
		this->lpOutputBuffer.resize(batchSize * this->GetBufferCount());

		// �덷�v�Z�p�̃o�b�t�@��������
		this->lpErrorValue_max.resize(this->GetBufferCount());
		this->lpErrorValue_ave.resize(this->GetBufferCount());
		this->lpErrorValue_ave2.resize(this->GetBufferCount());
		this->lpErrorValue_crossEntropy.resize(this->GetBufferCount());


		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessLearnLoop(const SettingData::Standard::IData& config)
	{
		return this->PreProcessCalculateLoop();
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessCalculateLoop()
	{
		this->calcErrorCount = 0;
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_max[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_ave[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_ave2[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_crossEntropy[0]),	0, sizeof(F32)*this->lpErrorValue_max.size());

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�b�`�T�C�Y���擾����.
		@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
	U32 IODataLayerGPU_base::GetBatchSize()const
	{
		return this->batchSize;
	}


	//==============================
	// ���͌n
	//==============================
	/** �w�K�덷���v�Z����.
		@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v */
	Gravisbell::ErrorCode IODataLayerGPU_base::CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		U32 inputBufferCount = this->GetInputBufferCount();

		if(this->lpDInputBuffer.size())
		{
			// �f�[�^���R�s�[
			this->lpDInputBuffer = this->lpOutputBuffer;

			// �f�[�^�̌덷���v�Z
			{
				float alpha = -1.0f;

				// y = alphat * x + y;
				cublasSaxpy(
					this->cublasHandle,
					inputBufferCount * this->batchSize,
					&alpha,
					i_lppInputBuffer,
					1,
					thrust::raw_pointer_cast(&this->lpDInputBuffer[0]),
					1);
			}
		}


		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			U32 bufferCount = this->GetBufferCount();
			dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_calculateError<<<grid, block>>>(
				i_lppInputBuffer,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_max[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_ave[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_ave2[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_crossEntropy[0]),
				batchNum,
				this->GetBufferCount());

			this->calcErrorCount++;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** �덷�̒l���擾����.
		CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
		@param	o_min	�ŏ��덷.
		@param	o_max	�ő�덷.
		@param	o_ave	���ό덷.
		@param	o_ave2	���ϓ��덷. */
	ErrorCode IODataLayerGPU_base::GetCalculateErrorValue(F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy)
	{
		o_max  = 0.0f;
		o_ave  = 0.0f;
		o_ave2 = 0.0f;
		o_crossEntropy = 0.0f;

		for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
		{
			F32 errorValue_max  = this->lpErrorValue_max[inputNum];
			F32 errorValue_ave  = this->lpErrorValue_ave[inputNum];
			F32 errorValue_ave2 = this->lpErrorValue_ave2[inputNum];
			F32 errorValue_crossEntropy = this->lpErrorValue_crossEntropy[inputNum];

			o_max   = max(o_max, errorValue_max);
			o_ave  += errorValue_ave;
			o_ave2 += errorValue_ave2;
			o_crossEntropy += errorValue_crossEntropy;
		}

		o_ave  = o_ave / this->calcErrorCount / this->GetBufferCount();
		o_ave2 = (F32)sqrt(o_ave2 / this->calcErrorCount / this->GetBufferCount());
		o_crossEntropy = o_crossEntropy / this->calcErrorCount / this->GetBufferCount();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �ڍׂȌ덷�̒l���擾����.
		�e���o�͂̒l���Ɍ덷�����.
		CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
		�e�z��̗v�f����[GetBufferCount()]�ȏ�ł���K�v������.
		@param	o_lpMin		�ŏ��덷.
		@param	o_lpMax		�ő�덷.
		@param	o_lpAve		���ό덷.
		@param	o_lpAve2	���ϓ��덷. */
	ErrorCode IODataLayerGPU_base::GetCalculateErrorValueDetail(F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[])
	{
		for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
		{
			o_lpMax[inputNum]   = this->lpErrorValue_max[inputNum];
			o_lpAve[inputNum]  += this->lpErrorValue_ave[inputNum] / this->GetDataCount();
			o_lpAve2[inputNum] += (F32)sqrt(this->lpErrorValue_ave2[inputNum] / this->GetDataCount());
		}
			
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct IODataLayerGPU_base::GetInputDataStruct()const
	{
		return this->GetDataStruct();
	}

	/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
	U32 IODataLayerGPU_base::GetInputBufferCount()const
	{
		return this->GetBufferCount();
	}

	/** �w�K�������擾����.
		�z��̗v�f����GetInputBufferCount�̖߂�l.
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER IODataLayerGPU_base::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** �w�K�������擾����.
		@param lpDOutputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	Gravisbell::ErrorCode IODataLayerGPU_base::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*batchSize*inputBufferCount, cudaMemcpyDeviceToHost);

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	//==============================
	// �o�͌n
	//==============================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct IODataLayerGPU_base::GetOutputDataStruct()const
	{
		return this->GetDataStruct();
	}

	/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
	U32 IODataLayerGPU_base::GetOutputBufferCount()const
	{
		return this->GetBufferCount();
	}

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER IODataLayerGPU_base::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	Gravisbell::ErrorCode IODataLayerGPU_base::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*batchSize*outputBufferCount, cudaMemcpyDeviceToHost);

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


}	// IOData
}	// Layer
}	// Gravisbell
