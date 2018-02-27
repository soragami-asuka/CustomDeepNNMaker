//===============================================
// 最適化ルーチン(SGD)
//===============================================

#include"Layer/NeuralNetwork/IOptimizer.h"

#include<string>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_base : public IOptimizer
	{
	public:
		static const std::wstring OPTIMIZER_ID;

	protected:
		U64 m_parameterCount;	/**< パラメータ数 */
		F32 m_learnCoeff;	/**< 学習係数 */

	public:
		/** コンストラクタ */
		Optimizer_SGD_base(U64 i_parameterCount);
		/** デストラクタ */
		virtual ~Optimizer_SGD_base();

	public:
		//===========================
		// 基本情報
		//===========================
		/** 識別IDの取得 */
		const wchar_t* GetOptimizerID()const;

		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		/** ハイパーパラメータを設定する
			@param	i_parameterID	パラメータ識別用ID
			@param	i_value			パラメータ. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);


	public:
		//===========================
		// 保存
		//===========================
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U64 GetUseBufferByteCount()const;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const;
	};


	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode ChangeOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U64 i_parameterCount);
	ErrorCode ChangeOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U64 i_parameterCount);

	/** オプティマイザをバッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_SGD(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize, IOptimizer* (*CreateOptimizer_SGD)(U64) );
	IOptimizer* CreateOptimizerFromBuffer_SGD_CPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize);
	IOptimizer* CreateOptimizerFromBuffer_SGD_GPU(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
