//=======================================
// NN���C���[�m�[�h�x�[�X
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
			NODETYPECODE_DATA_INPUT,	/**< ���̓f�[�^ */
			NODETYPECODE_DATA_OUTPUT,	/**< �o�̓f�[�^ */
			NODETYPECODE_DATA_TEACH,	/**< ���t�f�[�^ */

			NODETYPECODE_LAYER,
			NODETYPECODE_LAYER_BONDING,				/**< �����w */
			NODETYPECODE_LAYER_BONDING_ALL,			/**< �S�����w */
			NODETYPECODE_LAYER_BONDING_CONVOLUTION,	/**< ��݂��݌����w */
			NODETYPECODE_LAYER_ACTIVATION,			/**< �������w */
			NODETYPECODE_LAYER_POOLING,				/**< �v�[�����O�w */
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
		/** �R���X�g���N�^ */
		INNLayerBase(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerBase(){}

	public:
		/** �m�[�h��ʂ��擾���� */
		virtual NodeType.EnodeTypeCode GetNodeTypeCode()const=0;
		/** �m�[�h��ʂ̑������擾���� */
		virtual NodeType.ENodeTypeAttribute GetNoeTypeAttribute()const=0;
	}
}


#endif

