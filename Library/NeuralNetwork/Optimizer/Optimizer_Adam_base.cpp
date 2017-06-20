//===============================================
// 最適化ルーチン(Adam)
//===============================================
#include"stdafx.h"

#include"Optimizer_Adam_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	const std::wstring Optimizer_Adam_base::OPTIMIZER_ID = L"Adam";

	/** コンストラクタ */
	Optimizer_Adam_base::Optimizer_Adam_base(U32 i_parameterCount)
		:	m_parameterCount	(i_parameterCount)

		,	m_alpha			(0.001f)		/**< 慣性. */
		,	m_beta1			(0.9f)			/**< 減衰率. */
		,	m_beta2			(0.999f)		/**< 減衰率. */
		,	m_epsilon		(1e-8)			/**< 補助係数. */
	{
	}
	/** デストラクタ */
	Optimizer_Adam_base::~Optimizer_Adam_base()
	{
	}


	//===========================
	// 基本情報
	//===========================
	/** 識別IDの取得 */
	const wchar_t* Optimizer_Adam_base::GetOptimizerID()const
	{
		return OPTIMIZER_ID.c_str();
	}

	/** ハイパーパラメータを設定する
		@param	i_parameterID	パラメータ識別用ID
		@param	i_value			パラメータ. */
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		std::wstring parameter = i_parameterID;
		if(parameter == L"alpha")
		{
			this->m_alpha = i_value;
		}
		else if(parameter == L"beta1")
		{
			this->m_beta1 = i_value;
		}
		else if(parameter == L"beta2")
		{
			this->m_beta2 = i_value;
		}
		else if(parameter == L"epsilon")
		{
			this->m_epsilon = i_value;
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
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** ハイパーパラメータを設定する
		@param	i_parameterID	パラメータ識別用ID
		@param	i_value			パラメータ. */
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//===========================
	// 保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 Optimizer_Adam_base::GetUseBufferByteCount()const
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

		// α
		useBufferByte += sizeof(this->m_alpha);
		// β1
		useBufferByte += sizeof(this->m_beta1);
		// β2
		useBufferByte += sizeof(this->m_beta2);
		// ε
		useBufferByte += sizeof(this->m_epsilon);

		// M
		useBufferByte += sizeof(F32) * this->m_parameterCount;
		// V
		useBufferByte += sizeof(F32) * this->m_parameterCount;

		// β1^n
		useBufferByte += sizeof(F32);
		// β2^n
		useBufferByte += sizeof(F32);


		return useBufferByte;
	}

	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Optimizer_Adam_base::WriteToBufferBase(BYTE* o_lpBuffer)const
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


		// α
		memcpy(&o_lpBuffer[writePos], &this->m_alpha, sizeof(this->m_alpha));
		writePos+= sizeof(this->m_alpha);
		// β1
		memcpy(&o_lpBuffer[writePos], &this->m_beta1, sizeof(this->m_beta1));
		writePos+= sizeof(this->m_beta1);
		// β2
		memcpy(&o_lpBuffer[writePos], &this->m_beta2, sizeof(this->m_beta2));
		writePos+= sizeof(this->m_beta2);
		// ε
		memcpy(&o_lpBuffer[writePos], &this->m_epsilon, sizeof(this->m_epsilon));
		writePos+= sizeof(this->m_epsilon);


		return writePos;
	}

	/** バッファから作成する */
	Optimizer_Adam_base* CreateOptimizerFromBuffer_Adam(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize, Optimizer_Adam_base* (*CreateOptimizer_Adam)(U32) )
	{
		o_useBufferSize = -1;
		U32 readBufferPos = 0;

		// 使用バッファ数, IDは読み取り済み

		// パラメータ数
		U32 parameterCount = 0;
		memcpy(&parameterCount, &i_lpBuffer[readBufferPos], sizeof(parameterCount));
		readBufferPos += sizeof(parameterCount);

		// 作成
		Optimizer_Adam_base* pOptimizer = CreateOptimizer_Adam(parameterCount);
		if(pOptimizer == NULL)
			return NULL;


		// α
		F32 alpha = 0.0f;
		memcpy(&alpha, &i_lpBuffer[readBufferPos], sizeof(alpha));
		readBufferPos += sizeof(alpha);
		pOptimizer->SetHyperParameter(L"alpha", alpha);
		// β1
		F32 beta1 = 0.0f;
		memcpy(&beta1, &i_lpBuffer[readBufferPos], sizeof(beta1));
		readBufferPos += sizeof(beta1);
		pOptimizer->SetHyperParameter(L"beta1", beta1);
		// β2
		F32 beta2 = 0.0f;
		memcpy(&beta2, &i_lpBuffer[readBufferPos], sizeof(beta2));
		readBufferPos += sizeof(beta2);
		pOptimizer->SetHyperParameter(L"beta2", beta2);
		// ε
		F32 epsilon = 0.0f;
		memcpy(&epsilon, &i_lpBuffer[readBufferPos], sizeof(epsilon));
		readBufferPos += sizeof(epsilon);
		pOptimizer->SetHyperParameter(L"epsilon", epsilon);


		// 使用バッファ数保存
		o_useBufferSize = readBufferPos;

		return pOptimizer;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
