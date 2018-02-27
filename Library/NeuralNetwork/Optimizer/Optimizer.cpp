// Optimizer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"


#include"Library/NeuralNetwork/Optimizer.h"


#include"Optimizer_SGD_base.h"
#include"Optimizer_Adam_base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �I�v�e�B�}�C�U�[��ύX����.CPU */
	Optimizer_API ErrorCode ChangeOptimizer_CPU(IOptimizer** io_ppOptimizer, const wchar_t i_optimizerID[], U64 i_parameterCount)
	{
		std::wstring optimizerID = i_optimizerID;

		if(i_optimizerID == Optimizer_SGD_base::OPTIMIZER_ID)
		{
			ChangeOptimizer_SGD_CPU(io_ppOptimizer, i_parameterCount);
		}
		else if(i_optimizerID == Optimizer_Adam_base::OPTIMIZER_ID)
		{
			ChangeOptimizer_Adam_CPU(io_ppOptimizer, i_parameterCount);
		}
		else
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �I�v�e�B�}�C�U�[��ύX����.GPU */
	Optimizer_API ErrorCode ChangeOptimizer_GPU(IOptimizer** io_ppOptimizer, const wchar_t i_optimizerID[], U64 i_parameterCount)
	{
		std::wstring optimizerID = i_optimizerID;

		if(i_optimizerID == Optimizer_SGD_base::OPTIMIZER_ID)
		{
			ChangeOptimizer_SGD_GPU(io_ppOptimizer, i_parameterCount);
		}
		else if(i_optimizerID == Optimizer_Adam_base::OPTIMIZER_ID)
		{
			ChangeOptimizer_Adam_GPU(io_ppOptimizer, i_parameterCount);
		}
		else
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���ʕ����̓ǂݍ��� */
	ErrorCode ReadBaseFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize, U64& o_remainingBufferCount, std::wstring& o_optimizerID)
	{
		U64 readBufferPos = 0;

		// �g�p�o�b�t�@��
		U64 useBufferCount = 0;
		memcpy(&useBufferCount, &i_lpBuffer[readBufferPos], sizeof(useBufferCount));
		if(useBufferCount > i_bufferSize)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		readBufferPos += sizeof(useBufferCount);

		// ID�o�b�t�@��
		U32 idBufferSize = 0;
		memcpy(&idBufferSize, &i_lpBuffer[readBufferPos], sizeof(idBufferSize));
		readBufferPos += sizeof(idBufferSize);

		// ID
		wchar_t szID[256];
		memset(szID, 0, sizeof(szID));
		memcpy(szID, &i_lpBuffer[readBufferPos], idBufferSize);
		readBufferPos += idBufferSize;

		o_optimizerID = szID;
		o_useBufferSize = readBufferPos;
		o_remainingBufferCount = useBufferCount - readBufferPos;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �I�v�e�B�}�C�U�[���o�b�t�@����쐬����.CPU */
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_CPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
	{
		S64 readBufferPos = 0;
		U64 remainingBufferCount = 0;
		std::wstring optimizerID;

		if(ReadBaseFromBuffer(i_lpBuffer, i_bufferSize, readBufferPos, remainingBufferCount, optimizerID) != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		// �g���؂������Ƃɂ��Ă���
		o_useBufferSize = readBufferPos + remainingBufferCount;

		U64 useBufferSize = 0;
		if(optimizerID == Optimizer_SGD_base::OPTIMIZER_ID)
		{
			S64 useBufferCount = 0;
			return CreateOptimizerFromBuffer_SGD_CPU(&i_lpBuffer[readBufferPos], remainingBufferCount, useBufferCount);
		}
		else if(optimizerID == Optimizer_Adam_base::OPTIMIZER_ID)
		{
			S64 useBufferCount = 0;
			return CreateOptimizerFromBuffer_Adam_CPU(&i_lpBuffer[readBufferPos], remainingBufferCount, useBufferCount);
		}

		return NULL;
	}
	/** �I�v�e�B�}�C�U�[���o�b�t�@����쐬����.GPU */
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_GPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
	{
		S64 readBufferPos = 0;
		U64 remainingBufferCount = 0;
		std::wstring optimizerID;

		if(ReadBaseFromBuffer(i_lpBuffer, i_bufferSize, readBufferPos, remainingBufferCount, optimizerID) != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		// �g���؂������Ƃɂ��Ă���
		o_useBufferSize = readBufferPos + remainingBufferCount;

		U64 useBufferSize = 0;
		if(optimizerID == Optimizer_SGD_base::OPTIMIZER_ID)
		{
			S64 useBufferCount = 0;
			return CreateOptimizerFromBuffer_SGD_GPU(&i_lpBuffer[readBufferPos], remainingBufferCount, useBufferCount);
		}
		else if(optimizerID == Optimizer_Adam_base::OPTIMIZER_ID)
		{
			S64 useBufferCount = 0;
			return CreateOptimizerFromBuffer_Adam_GPU(&i_lpBuffer[readBufferPos], remainingBufferCount, useBufferCount);
		}

		return NULL;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
