//=======================================
// レイヤーベース
//=======================================
#ifndef __I_NN_LAYER_BASE_H__
#define __I_NN_LAYER_BASE_H__

#include"IInputLayer.h"
#include"IOutputLayer.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class INNLayer : public IInputLayer, public virtual IOutputLayer
	{
	public:
		/** コンストラクタ */
		INNLayer(){}
		/** デストラクタ */
		virtual ~INNLayer(){}

	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ELayerErrorCode Initialize(void) = 0;
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ELayerErrorCode Initialize(const INNLayerConfig& config) = 0;

		/** 設定情報を設定 */
		virtual ELayerErrorCode SetLayerConfig(const INNLayerConfig& config) = 0;
		/** レイヤーの設定情報を取得する */
		virtual const INNLayerConfig* GetLayerConfig()const = 0;

		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** レイヤーを読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;

	public:
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		virtual ELayerErrorCode Calculate() = 0;

		/** 学習差分をレイヤーに反映させる */
		virtual ELayerErrorCode ReflectionLearnError(void) = 0;
	};
}

#endif