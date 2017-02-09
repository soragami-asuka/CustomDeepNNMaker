//=======================================
// レイヤーベース
//=======================================
#ifndef __LAYER_ERROR_CODE_H__
#define __LAYER_ERROR_CODE_H__

namespace CustomDeepNNLibrary
{
	enum ELayerErrorCode
	{
		LAYER_ERROR_NONE = 0,

		// 共通系エラー
		LAYER_ERROR_COMMON = 0x01000000,
		LAYER_ERROR_COMMON_NULL_REFERENCE,		///< NULL参照
		LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE,	///< 配列外参照
		LAYER_ERROR_COMMON_OUT_OF_VALUERANGE,	///< 値範囲の外
		LAYER_ERROR_COMMON_ALLOCATION_MEMORY,	///< メモリの確保に失敗
		LAYER_ERROR_COMMON_FILE_NOT_FOUND,		///< ファイルの参照に失敗

		// レイヤー系エラー
		LAYER_ERROR_LAYER = 0x02000000,
		LAYER_ERROR_NONREGIST_CONFIG,			///< 設定情報が登録されていない
		LAYER_ERROR_FRAUD_INPUT_COUNT,			///< 入力数が不正
		LAYER_ERROR_FRAUD_OUTPUT_COUNT,			///< 出力数が不正
		LAYER_ERROR_FRAUD_NEURON_COUNT,			///< ニューロン数が不正
		// レイヤー追加系エラー
		LAYER_ERROR_ADDLAYER = 0x02010000,
		LAYER_ERROR_ADDLAYER_ALREADY_SAMEID,	///< 同じIDのレイヤーが既に登録済み
		// レイヤー削除系エラー
		LAYER_ERROR_ERASELAYER = 0x02020000,
		LAYER_ERROR_ERASELAYER_NOTFOUND,		///< 対象のID検索に失敗
		// レイヤー初期化系エラー
		LAYER_ERROR_INITLAYER_DISAGREE_CONFIG,	///< 設定情報の型が不一致
		LAYER_ERROR_INITLAYER_READ_CONFIG,		///< 設定情報の読み取りに失敗

		// DLL系エラー
		LAYER_ERROR_DLL = 0x03000000,
		LAYER_ERROR_DLL_LOAD_FUNCTION,		///< 関数の読み込みに失敗
		LAYER_ERROR_DLL_ADD_ALREADY_SAMEID,	///< 既に同一IDのDLLが登録済み
		LAYER_ERROR_DLL_ERASE_NOTFOUND,		///< 対象のID検索に失敗

		// 入出力系エラー
		LAYER_ERROR_IO = 0x04000000,
		LAYER_ERROR_IO_DISAGREE_INPUT_OUTPUT_COUNT,	///< 入出力のデータ数が一致しない


		// ユーザー定義
		LAYER_ERROR_USER = 0x70000000
	};
}

#endif