//=======================================
// ニューラルネットワークのレイヤーに関するデータを取り扱うインターフェース
// バッファなどを管理する.
//=======================================
#ifndef __GRAVISBELL_I_LAYER_CONNECT_DATA_H__
#define __GRAVISBELL_I_LAYER_CONNECT_DATA_H__

#include"../ILayerData.h"
#include"../NeuralNetwork/INeuralNetwork.h"

namespace Gravisbell {
namespace Layer {
namespace Connect {

	class ILayerConnectData : public ILayerData
	{
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		ILayerConnectData() : ILayerData(){}
		/** デストラクタ */
		virtual ~ILayerConnectData(){}

		//===========================
		// レイヤー構造
		//===========================
		/** 入力データ構造が使用可能か確認する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	使用可能な入力データ構造の場合trueが返る. */
		virtual bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		virtual bool CheckCanUseInputDataStruct(Gravisbell::GUID i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;

		/** 出力データ構造を取得する.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
		virtual IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		virtual IODataStruct GetOutputDataStruct(Gravisbell::GUID i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;


		//====================================
		// レイヤーの追加/削除/管理
		//====================================
	public:
		/** レイヤーデータを追加する.
			@param	i_guid			追加するレイヤーに割り当てられるGUID.
			@param	i_pLayerData	追加するレイヤーデータのアドレス.
			@param	i_onFixFlag		レイヤーを固定化するフラグ. */
		virtual ErrorCode AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData, bool i_onFixFlag) = 0;
		/** レイヤーデータを削除する.
			@param i_guid	削除するレイヤーのGUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& i_guid) = 0;
		/** レイヤーデータを全削除する */
		virtual ErrorCode EraseAllLayer() = 0;

		/** 登録されているレイヤー数を取得する */
		virtual U32 GetLayerCount() = 0;
		/** レイヤーのGUIDを番号指定で取得する */
		virtual ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid) = 0;

		/** 登録されているレイヤーデータを番号指定で取得する */
		virtual ILayerData* GetLayerDataByNum(U32 i_layerNum) = 0;
		/** 登録されているレイヤーデータをGUID指定で取得する */
		virtual ILayerData* GetLayerDataByGUID(const Gravisbell::GUID& i_guid) = 0;

		/** レイヤーの固定化フラグを取得する */
		virtual bool GetLayerFixFlagByGUID(const Gravisbell::GUID& i_guid) = 0;

		//====================================
		// 入出力レイヤー
		//====================================
	public:
		/** 入力信号に割り当てられているGUIDを取得する */
		virtual GUID GetInputGUID() = 0;

		/** 出力信号レイヤーを設定する */
		virtual ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid) = 0;
		/** 出力信号レイヤーのGUIDを取得する */
		virtual Gravisbell::GUID GetOutputLayerGUID() = 0;


		//====================================
		// レイヤーの接続
		//====================================
	public:
		/** レイヤーに入力レイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** レイヤーにバイパスレイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** レイヤーから入力レイヤーを削除する. 
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode EraseInputLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** レイヤーからバイパスレイヤーを削除する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode EraseBypassLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID) = 0;
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID) = 0;

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID) = 0;
		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID) = 0;

		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		virtual ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID) = 0;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		virtual ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID) = 0;

	public:
		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager) = 0;
		/** レイヤーを作成する.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@param	i_useHostMemory		内部的にホストメモリを使用する. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager, bool i_useHostMemory) = 0;

		/** レイヤーを作成する.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		/** レイヤーを作成する.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
			@param	i_useHostMemory		内部的にホストメモリを使用する. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, bool i_useHostMemory) = 0;
	};

}	// Connect
}	// Layer
}	// Gravisbell

#endif