// Optimizer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"


#include"Library/NeuralNetwork/Optimizer.h"


#include"Optimizer_SGD_base.h"
#include"Optimizer_Adam_base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** オプティマイザーを変更する.CPU */
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
	/** オプティマイザーを変更する.GPU */
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

	/** 共通部分の読み込み */
	ErrorCode ReadBaseFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize, U64& o_remainingBufferCount, std::wstring& o_optimizerID)
	{
		U64 readBufferPos = 0;

		// 使用バッファ数
		U64 useBufferCount = 0;
		memcpy(&useBufferCount, &i_lpBuffer[readBufferPos], sizeof(useBufferCount));
		if(useBufferCount > i_bufferSize)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		readBufferPos += sizeof(useBufferCount);

		// IDバッファ数
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

	/** オプティマイザーをバッファから作成する.CPU */
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_CPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
	{
		S64 readBufferPos = 0;
		U64 remainingBufferCount = 0;
		std::wstring optimizerID;

		if(ReadBaseFromBuffer(i_lpBuffer, i_bufferSize, readBufferPos, remainingBufferCount, optimizerID) != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		// 使い切ったことにしておく
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
	/** オプティマイザーをバッファから作成する.GPU */
	Optimizer_API IOptimizer* CreateOptimizerFromBuffer_GPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
	{
		S64 readBufferPos = 0;
		U64 remainingBufferCount = 0;
		std::wstring optimizerID;

		if(ReadBaseFromBuffer(i_lpBuffer, i_bufferSize, readBufferPos, remainingBufferCount, optimizerID) != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		// 使い切ったことにしておく
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
