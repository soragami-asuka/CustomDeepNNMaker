//=======================================
// ニューラルネットワーク本体定義
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNLayer
	{
	public:
		/** コンストラクタ */
		INeuralNetwork(){}
		/** デストラクタ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// レイヤーの追加
		//====================================
		/** レイヤーを追加する.
			追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param pLayer	追加するレイヤーのアドレス. */
		virtual ErrorCode AddLayer(INNLayer* pLayer) = 0;
		/** レイヤーを削除する.
			@param i_guid	削除するレイヤーのGUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& i_guid) = 0;
		/** レイヤーを全削除する */
		virtual ErrorCode EraseAllLayer() = 0;

		/** 登録されているレイヤー数を取得する */
		virtual ErrorCode GetLayerCount()const = 0;
		/** レイヤーを番号指定で取得する */
		virtual const INNLayer* GetLayerByNum(const U32 i_layerNum) = 0;
		/** レイヤーをGUID指定で取得する */
		virtual const INNLayer* GetLayerByGUID(const Gravisbell::GUID& i_guid) = 0;


		//====================================
		// 入出力レイヤー
		//====================================
		/** 入力信号に割り当てられているGUIDを取得する */
		virtual GUID GetInputGUID()const = 0;

		/** 出力信号に割り当てらているレイヤーのGUIDを取得する */
		virtual GUID GetOutputLayerGUID()const = 0;
		/** 出力信号レイヤーを設定する */
		virtual GUID SetOutputLayerGUID(const Gravisbell::GUID& i_guid) = 0;


		//====================================
		// レイヤーの接続
		//====================================
		/** レイヤーに入力レイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** レイヤーにバイパスレイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		virtual ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID) = 0;
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID) = 0;

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const = 0;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		virtual ErrorCode GetInputLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const = 0;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const = 0;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		virtual ErrorCode GetBypassLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const = 0;


		/** レイヤーの接続状態に異常がないかチェックする.
			@param	o_errorLayer	エラーが発生したレイヤーGUID格納先. 
			@return	接続に異常がない場合はNO_ERROR, 異常があった場合は異常内容を返し、対象レイヤーのGUIDをo_errorLayerに格納する. */
		virtual ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer) = 0;



		//====================================
		// 学習設定
		//====================================
		/** 学習設定を取得する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID. */
		virtual const SettingData::Standard::IData* GetLearnSettingData(const Gravisbell::GUID& guid) = 0;

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID
			@param	data	設定する学習設定 */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& data) = 0;

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
