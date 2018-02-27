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
#include"../Common/ITemporaryMemoryManager.h"

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
		virtual U64 GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;


	public:
		//===========================
		// レイヤー構造
		//===========================
		/** 入力データ構造が使用可能か確認する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	使用可能な入力データ構造の場合trueが返る. */
		virtual bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;

		/** 出力データ構造を取得する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
		virtual IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;

		/** 複数出力が可能かを確認する */
		virtual bool CheckCanHaveMultOutputLayer(void) = 0;

	public:
		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager) = 0;

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