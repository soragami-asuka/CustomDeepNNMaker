//=======================================
// レイヤーに関するデータを取り扱うインターフェース
// バッファなどを管理する.
//=======================================
#ifndef __GRAVISBELL_I_LAYER_DATA_H__
#define __GRAVISBELL_I_LAYER_DATA_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

#include"./ILayerBase.h"

namespace Gravisbell {
namespace Layer {

	class ILayerData
	{
	public:
		/** コンストラクタ */
		ILayerData(){}
		/** デストラクタ */
		virtual ~ILayerData(){}

		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		virtual ErrorCode Initialize(void) = 0;


		//===========================
		// 共通制御
		//===========================
	public:
		/** レイヤー固有のGUIDを取得する */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;

		/** レイヤーの設定情報を取得する */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual U32 GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S32 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		
	public:
		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid) = 0;

	public:
		//===========================
		// オプティマイザー設定
		//===========================
		/** オプティマイザーを変更する */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** オプティマイザーのハイパーパラメータを変更する */
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]) = 0;
	};

}	// Layer
}	// Gravisbell

#endif