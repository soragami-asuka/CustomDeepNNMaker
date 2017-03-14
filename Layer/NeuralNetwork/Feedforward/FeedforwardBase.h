//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"Feedforward_DATA.hpp"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class FeedforwardBase : public Gravisbell::Layer::NeuralNetwork::INNLayer
	{
	protected:
		GUID guid;	/**< ���C���[���ʗp��GUID */

		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		SettingData::Standard::IData* pLearnData;		/**< �w�K�ݒ���`�����R���t�B�O�N���X */

		Feedforward::LayerStructure layerStructure;	/**< ���C���[�\�� */
		Feedforward::LearnDataStructure learnData;	/**< �w�K�ݒ� */

		unsigned int batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		FeedforwardBase(GUID guid);

		/** �f�X�g���N�^ */
		virtual ~FeedforwardBase();

		//===========================
		// ���C���[����
		//===========================
	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		unsigned int GetLayerKindBase()const;

		/** ���C���[�ŗL��GUID���擾���� */
		ErrorCode GetGUID(GUID& o_guid)const;

		/** ���C���[�̎�ގ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ErrorCode GetLayerCode(GUID& o_layerCode)const;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		unsigned int GetBatchSize()const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);
		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerConfig()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		unsigned int GetUseBufferByteCount()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const;


		//===========================
		// �ŗL�֐�
		//===========================
	public:
		/** �j���[���������擾���� */
		unsigned int GetNeuronCount()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell