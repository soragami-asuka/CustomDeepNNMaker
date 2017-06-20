//===============================================
// 最適化ルーチン(SGD)
//===============================================
#include"stdafx.h"

#include"Optimizer_SGD_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	const std::wstring Optimizer_SGD_base::OPTIMIZER_ID = L"SGD";

	/** コンストラクタ */
	Optimizer_SGD_base::Optimizer_SGD_base(U32 i_parameterCount)
		:	m_parameterCount	(i_parameterCount)
		,	m_learnCoeff		(1.0f)
	{
	}
	/** デストラクタ */
	Optimizer_SGD_base::~Optimizer_SGD_base()
	{
	}


	//===========================
	// 基本情報
	//===========================
	/** 識別IDの取得 */
	const wchar_t* Optimizer_SGD_base::GetOptimizerID()const
	{
		return OPTIMIZER_ID.c_str();
	}

	/** ハイパーパラメータを設定する
		@param	i_parameterID	パラメータ識別用ID
		@param	i_value			パラメータ. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		std::wstring parameter = i_parameterID;
		if(parameter == L"LearnCoeff")
		{
			this->m_learnCoeff = i_value;
		}
		else
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ハイパーパラメータを設定する
		@param	i_parameterID	パラメータ識別用ID
		@param	i_value			パラメータ. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** ハイパーパラメータを設定する
		@param	i_parameterID	パラメータ識別用ID
		@param	i_value			パラメータ. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//===========================
	// 保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 Optimizer_SGD_base::GetUseBufferByteCount()const
	{
		U32 useBufferByte = 0;

		// 使用バイト数格納
		useBufferByte += sizeof(U32);

		// IDバッファサイズ
		useBufferByte += sizeof(U32);

		// IDバッファ
		useBufferByte += sizeof(wchar_t) * OPTIMIZER_ID.size();

		// パラメータ数
		useBufferByte += sizeof(this->m_parameterCount);

		// 学習係数
		useBufferByte += sizeof(this->m_learnCoeff);

		return useBufferByte;
	}

	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Optimizer_SGD_base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		U32 writePos = 0;

		// 使用バイト数
		U32 userBufferByte = this->GetUseBufferByteCount();
		memcpy(&o_lpBuffer[writePos], &userBufferByte, sizeof(userBufferByte));
		writePos += sizeof(userBufferByte);

		// IDバッファサイズ
		U32 idBufferSize = sizeof(wchar_t) * OPTIMIZER_ID.size();
		memcpy(&o_lpBuffer[writePos], &idBufferSize, sizeof(idBufferSize));
		writePos += sizeof(idBufferSize);

		// IDバッファ
		memcpy(&o_lpBuffer[writePos], (const BYTE*)OPTIMIZER_ID.c_str(), idBufferSize);
		writePos += idBufferSize;

		// パラメータ数
		memcpy(&o_lpBuffer[writePos], &this->m_parameterCount, sizeof(this->m_parameterCount));
		writePos+= sizeof(this->m_parameterCount);

		// 学習係数
		memcpy(&o_lpBuffer[writePos], &this->m_learnCoeff, sizeof(this->m_learnCoeff));
		writePos+= sizeof(this->m_learnCoeff);

		return writePos;
	}

	/** バッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_SGD(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize, IOptimizer* (*CreateOptimizer_SGD)(U32) )
	{
		o_useBufferSize = -1;
		U32 readBufferPos = 0;

		// 使用バッファ数, IDは読み取り済み

		// パラメータ数
		U32 parameterCount = 0;
		memcpy(&parameterCount, &i_lpBuffer[readBufferPos], sizeof(parameterCount));
		readBufferPos += sizeof(parameterCount);

		// 作成
		IOptimizer* pOptimizer = CreateOptimizer_SGD(parameterCount);
		if(pOptimizer == NULL)
			return NULL;

		// 学習係数
		F32 learnCoeff = 0.0f;
		memcpy(&learnCoeff, &i_lpBuffer[readBufferPos], sizeof(learnCoeff));
		readBufferPos += sizeof(learnCoeff);
		pOptimizer->SetHyperParameter(L"LearnCoeff", learnCoeff);

		// 使用バッファ数保存
		o_useBufferSize = readBufferPos;

		return pOptimizer;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
