//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// GPU����
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_base.cuh"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Library/Common/TemporaryMemoryManager.h"

// CUDA�p
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda_runtime_api.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	static std::wstring GetInputTemporaryBufferID(U32 inputNum)
	{
		wchar_t szBuf[32];
		swprintf(szBuf, L"input[%d]", inputNum);

		return szBuf;
	}
	static std::wstring GetDInputTemporaryBufferID(U32 inputNum)
	{
		wchar_t szBuf[32];
		swprintf(szBuf, L"dinput[%d]", inputNum);

		return szBuf;
	}


	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_base::FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount, i_layerData.GetOutputDataStruct(i_lpInputDataStruct, i_inputLayerCount), Common::CreateTemporaryMemoryManagerGPU())
	{
	}
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_base::FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount, i_layerData.GetOutputDataStruct(i_lpInputDataStruct, i_inputLayerCount), i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_GPU_base::~FeedforwardNeuralNetwork_GPU_base()
	{
	}


	//====================================
	// ���͌덷�o�b�t�@�֘A
	//====================================
	/** ���͌덷�o�b�t�@�̑�����ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::SetDInputBufferCount(U32 i_DInputBufferCount)
	{
		this->lpDInputBuffer.resize(i_DInputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���͌덷�o�b�t�@�̃T�C�Y��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBuffer[i_DInputBufferNo].resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���͌덷�o�b�t�@���擾���� */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetTmpDInputBuffer_d(U32 i_DInputBufferNo)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}
	/** ���͌덷�o�b�t�@���擾���� */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetTmpDInputBuffer_d(U32 i_DInputBufferNo)const
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}


	//====================================
	// ���o�̓o�b�t�@�֘A
	//====================================
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer_h[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::PreProcessLearn(U32 batchSize)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::PreProcessLearn(batchSize);

		for(U32 i=0; i<this->GetInputCount(); i++)
		{
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), GetInputTemporaryBufferID(i).c_str(), sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), GetDInputTemporaryBufferID(i).c_str(), sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize());
		}
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());

		// �o�̓o�b�t�@�̊m��
		this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

		return err;
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::PreProcessCalculate(unsigned int batchSize)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::PreProcessCalculate(batchSize);
		
		for(U32 i=0; i<this->GetInputCount(); i++)
		{
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), GetInputTemporaryBufferID(i).c_str(),  sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize());
		}
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());

		// �o�̓o�b�t�@�̊m��
		this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

		return err;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

		// �o�̓o�b�t�@���R�s�[
		if(err == ErrorCode::ERROR_CODE_NONE)
		{
			if(o_lppOutputBuffer)
			{
				cudaMemcpy(o_lppOutputBuffer, this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			}
			cudaMemcpy(&this->lpOutputBuffer_h[0], this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
		}

		return err;
	}
	/** ���Z���������s����.
		�z�X�g���������n�����
		@param i_lppInputBuffer		���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
		@param o_lppOutputBuffer	�o�̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ���̓o�b�t�@���R�s�[
		std::vector<const F32*> lppInputBuffer(this->GetInputCount());
		for(U32 i=0; i<this->GetInputCount(); i++)
		{
			if(this->lppInputTmpBuffer[i].empty())
			{
				this->lppInputTmpBuffer[i].resize(this->GetInputBufferCount(i) * this->GetBatchSize());
				this->lppInputBuffer[i] = &this->lppInputTmpBuffer[i][0];
			}
			memcpy(&this->lppInputTmpBuffer[i][0], i_lppInputBuffer[i], sizeof(F32)*this->lppInputTmpBuffer[i].size());

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lpInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());
			cudaMemcpy(lpInputBuffer, &this->lppInputTmpBuffer[i][0], sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize(), cudaMemcpyHostToDevice);

			lppInputBuffer[i] = lpInputBuffer;
		}

		// ���Z
		Gravisbell::ErrorCode err = this->Calculate_device(&lppInputBuffer[0], NULL);

		if(o_lppOutputBuffer)
			this->GetOutputBuffer(o_lppOutputBuffer);

		// �o�b�t�@���J��
		for(U32 i=0; i<this->GetInputCount(); i++)
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());

		return err;
	}
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer[])
	{
		return this->Calculate(i_lpInputBuffer, NULL);
	}

	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
		std::vector<const F32*> lppInputBuffer(this->GetInputCount());
		for(U32 i=0; i<this->GetInputCount(); i++)
		{
			F32* lpInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());
			cudaMemcpy(lpInputBuffer, &this->lppInputTmpBuffer[i][0], sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize(), cudaMemcpyHostToDevice);

			lppInputBuffer[i] = lpInputBuffer;
		}

		// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
		F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
		cudaMemcpy(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		Gravisbell::ErrorCode err;
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@���m��
			std::vector<F32*> lppDInputBuffer(this->GetInputCount(), NULL);
			for(U32 i=0; i<this->GetInputCount(); i++)
			{
				lppDInputBuffer[i] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), GetDInputTemporaryBufferID(i).c_str());
			}

			// ���Z
			err = CalculateDInput_device(&lppInputBuffer[0], &lppDInputBuffer[0], NULL, lppDOutputBuffer);

			// ���͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			for(U32 i=0; i<this->GetInputCount(); i++)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer[i], sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), GetDInputTemporaryBufferID(i).c_str());
			}
		}
		else
		{
			// ���Z
			err = CalculateDInput_device(&lppInputBuffer[0], NULL, NULL, lppDOutputBuffer);
		}

		// �o�b�t�@���J��
		for(U32 i=0; i<this->GetInputCount(); i++)
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

		return err;
	}
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(NULL, o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}



	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
		std::vector<const F32*> lppInputBuffer(this->GetInputCount());
		for(U32 i=0; i<this->GetInputCount(); i++)
		{
			F32* lpInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());
			cudaMemcpy(lpInputBuffer, &this->lppInputTmpBuffer[i][0], sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize(), cudaMemcpyHostToDevice);

			lppInputBuffer[i] = lpInputBuffer;
		}

		// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
		F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
		cudaMemcpy(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);


		Gravisbell::ErrorCode err;
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@���m��
			std::vector<F32*> lppDInputBuffer(this->GetInputCount(), NULL);
			for(U32 i=0; i<this->GetInputCount(); i++)
			{
				lppDInputBuffer[i] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), GetDInputTemporaryBufferID(i).c_str());
			}

			// ���Z
			err = Training_device(&lppInputBuffer[0], &lppDInputBuffer[0], NULL, lppDOutputBuffer);

			// ���͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			for(U32 i=0; i<this->GetInputCount(); i++)
			{
				cudaMemcpy(o_lppDInputBuffer[i], lppDInputBuffer[i], sizeof(F32)*this->GetInputBufferCount(i)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), GetDInputTemporaryBufferID(i).c_str());
			}
		}
		else
		{
			// ���Z
			err = Training_device(&lppInputBuffer[0], NULL, NULL, lppDOutputBuffer);
		}

		// �o�b�t�@���J��
		for(U32 i=0; i<this->GetInputCount(); i++)
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), GetInputTemporaryBufferID(i).c_str());
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

		return err;
	}
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->Training(NULL, o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

