//=======================================
// NNレイヤーノードベース
//=======================================
#ifndef __I_NN_NODE_BASE_H__
#define __I_NN_NODE_BASE_H__

namespace CustomDeepNNLibrary
{
	namespace NodeType
	{
		enum ENodeTypeCode
		{
			NODETYPECODE_DATA,
			NODETYPECODE_DATA_INPUT,	/**< 入力データ */
			NODETYPECODE_DATA_OUTPUT,	/**< 出力データ */
			NODETYPECODE_DATA_TEACH,	/**< 教師データ */

			NODETYPECODE_LAYER,
			NODETYPECODE_LAYER_BONDING,				/**< 結合層 */
			NODETYPECODE_LAYER_BONDING_ALL,			/**< 全結合層 */
			NODETYPECODE_LAYER_BONDING_CONVOLUTION,	/**< 畳みこみ結合層 */
			NODETYPECODE_LAYER_ACTIVATION,			/**< 活性化層 */
			NODETYPECODE_LAYER_POOLING,				/**< プーリング層 */
		};
		enum ENodeTypeAttribute
		{
			NODETYPEATTR_DATA        = (0x01 << NODETYPECODE_DATA);
			NODETYPEATTR_DATA_INPUT  = NODETYPEATTR_DATA | (0x01 << NODETYPECODE_DATA_INPUT),
			NODETYPEATTR_DATA_OUTPUT = NODETYPEATTR_DATA | (0x01 << NODETYPECODE_DATA_OUTPUT),
			NODETYPEATTR_DATA_TEACH  = NODETYPEATTR_DATA | (0x01 << NODETYPECODE_DATA_TEACH),

			NODETYPEATTR_LAYER                     = (0x01 << NODETYPECODE_LAYER),
			NODETYPECODE_LAYER_BONDING             = NODETYPEATTR_LAYER            | (0x01 << NODETYPECODE_LAYER_CONCATRATE),
			NODETYPECODE_LAYER_BONDING_ALL         = NODETYPECODE_LAYER_BONDING    | (0x01 << NODETYPECODE_LAYER_CONCATRATE_ALL),
			NODETYPECODE_LAYER_BONDING_CONVOLUTION = NODETYPECODE_LAYER_BONDING    | (0x01 << NODETYPECODE_LAYER_CONCATRATE_CONVOLUTION),
			NODETYPECODE_LAYER_ACTIVATION          = NODETYPECODE_LAYER            | (0x01 << NODETYPECODE_LAYER_ACTIVATION),
			NODETYPECODE_LAYER_POOLING             = NODETYPECODE_LAYER            | (0x01 << NODETYPECODE_LAYER_POOLING),
		};
	}

	class INNNodeBase
	{
	public:
		/** コンストラクタ */
		INNLayerBase(){}
		/** デストラクタ */
		virtual ~INNLayerBase(){}

	public:
		/** ノード種別を取得する */
		virtual NodeType.EnodeTypeCode GetNodeTypeCode()const=0;
		/** ノード種別の属性を取得する */
		virtual NodeType.ENodeTypeAttribute GetNoeTypeAttribute()const=0;
	}
}


#endif

